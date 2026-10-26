"""
engine.py - DJAYAIIII core audio engine
=======================================

A zero-dependency audio toolkit written with nothing but the Python standard
library (wave, math, struct, array, io, operator, itertools).  There is
nothing to pip-install: if Python 3.8+ runs, this runs.

Speed: every per-sample operation is pushed down into C-level builtins
(array, map, operator, slicing, struct).  Instruments are rendered once as
short templates and tiled across the track, so a full-length remix renders
in a few seconds instead of freezing the process in a Python for-loop.

Two layers live here:

  * A tiny synth (kick + square sub-bass) and a tech-house remixer that
    works fully offline - used when no generative-AI key is configured.
  * Stem tools used by app.py to assemble AI-generated instrumentals with
    the isolated vocals of any song: decoding, resampling, seam-joining,
    transient ducking, stem mixing and mastering.

Public API
----------
generate_tech_house_backing(duration_secs, bpm=126, sample_rate=44100)
    -> array('d') of mono float samples in [-1.0, 1.0]

remix_track(vocal_wav_bytes, bpm=126, sample_rate=44100, source_bpm=None,
            intro_bars=2, outro_bars=2)
    -> bytes of a finished 16-bit stereo WAV (offline tech-house remix)

decode_wav(wav_bytes, mono=True)
    -> (array('d') | [array('d') per channel], sample_rate)

resample(samples, src_rate, dst_rate)        -> array('d')
fit_length(samples, n)                       -> array('d') trimmed / zero-padded
butt_join(chunks, sample_rate, fade_secs)    -> array('d') click-free concatenation
transient_duck_envelope(trigger, sample_rate, ...) -> array('d') gain curve
mix_stems(vocal_channels, inst_channels, sample_rate, ...) -> [array('d'), ...]

samples_to_wav_bytes(samples, sample_rate=44100, stereo=True)
    -> bytes   float samples (mono array, or [left, right]) -> mastered WAV

render_backing_wav(duration_secs, bpm=126, sample_rate=44100)
    -> bytes   quick preview of just the synth backing track.
"""

import io
import math
import operator
import struct
import sys
import wave
from array import array
from itertools import repeat

TWO_PI = 2.0 * math.pi

# ---------------------------------------------------------------------------
# Sound-design knobs
# ---------------------------------------------------------------------------

# Kick: a sine whose pitch slides 150Hz -> 40Hz, soft-clipped for punch.
KICK_START_HZ = 150.0
KICK_END_HZ = 40.0
KICK_PITCH_DECAY = 0.035     # seconds - how fast the pitch slide happens
KICK_AMP_DECAY = 0.11        # seconds - body decay time
KICK_LENGTH = 0.40           # seconds - total length of one kick hit
KICK_DRIVE = 1.6             # tanh drive; >1 adds click and weight
KICK_GAIN = 0.95

# Sub-bass: gated square wave on the off-beats ("and" of every beat).
# One MIDI note per off-beat, looping over two bars (8 beats).
# 29 = F1 (43.65Hz)   32 = Ab1 (51.91Hz)   27 = Eb1 (38.89Hz)
BASS_PATTERN_MIDI = [29, 29, 29, 29, 29, 29, 32, 27]
BASS_GATE = 0.42             # fraction of a beat each note sustains
BASS_ATTACK = 0.004          # seconds
BASS_RELEASE = 0.030         # seconds
BASS_LOWPASS_HZ = 240.0      # one-pole low-pass keeps it sub, not buzzy
BASS_GAIN = 0.55

# Sidechain: the vocal is pinned to 25% on every kick, then swells back.
SIDECHAIN_FLOOR = 0.25
SIDECHAIN_HOLD = 0.04        # seconds held at the floor after each kick
SIDECHAIN_RELEASE = 0.26     # seconds to swell back to full volume

VOCAL_GAIN = 0.85            # vocal level (after peak normalisation)
MASTER_PEAK = 0.95           # final peak level of the mastered mix
MAX_GRID_STRETCH = 0.25      # never stretch a vocal more than +/-25%

# Transient ducking (used when laying vocals over an AI instrumental):
# every hit in the instrumental pulls the vocal down to DUCK_FLOOR for a
# moment, then it swells back - genre-agnostic "glue".
DUCK_FLOOR = 0.5
DUCK_HOLD = 0.03             # seconds
DUCK_RELEASE = 0.18          # seconds
DUCK_BLOCK = 1024            # samples per analysis block (~23ms at 44.1k)
DUCK_ONSET_RATIO = 2.2       # block must be this many times the running average
DUCK_MIN_LEVEL = 0.02        # ignore "hits" quieter than this (mean |x|)
DUCK_MIN_GAP = 0.08          # seconds between two detected hits

_CHUNK = 1 << 20             # samples per block in the resampler


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _midi_hz(note):
    """MIDI note number -> frequency in Hz (A4 = 69 = 440Hz)."""
    return 440.0 * 2.0 ** ((note - 69) / 12.0)


def _samples_per_beat(bpm, sample_rate):
    return max(1, int(round(60.0 / bpm * sample_rate)))


def _add_into(dst, src, start):
    """dst[start:start+len(src)] += src   (clipped to the end of dst)."""
    end = min(len(dst), start + len(src))
    if end <= start:
        return
    seg = src[: end - start]
    dst[start:end] = array("d", map(operator.add, dst[start:end], seg))


def _tile(template, total):
    """Repeat a template array until it is exactly `total` samples long."""
    reps = total // len(template) + 1
    return (template * reps)[:total]


def _native(arr):
    """array -> bytes in little-endian order regardless of host CPU."""
    if sys.byteorder == "big":
        arr = array(arr.typecode, arr)
        arr.byteswap()
    return arr.tobytes()


def _int_array(typecode, raw):
    """bytes (little-endian PCM) -> array of ints, C-speed."""
    arr = array(typecode)
    arr.frombytes(raw)
    if sys.byteorder == "big":
        arr.byteswap()
    return arr


# ---------------------------------------------------------------------------
# Instrument rendering (short one-shot templates, rendered once)
# ---------------------------------------------------------------------------

def _render_kick(sample_rate):
    """One punchy kick: sine sliding 150Hz -> 40Hz with a fast decay."""
    n = int(KICK_LENGTH * sample_rate)
    fade = min(200, n)
    sweep = KICK_START_HZ - KICK_END_HZ
    out = array("d")
    for i in range(n):
        t = i / sample_rate
        # Frequency f(t) = END + SWEEP * e^(-t/tau).  Integrate f(t) to get
        # phase so the slide is click-free and perfectly continuous.
        phase = TWO_PI * (
            KICK_END_HZ * t
            + sweep * KICK_PITCH_DECAY * (1.0 - math.exp(-t / KICK_PITCH_DECAY))
        )
        env = math.exp(-t / KICK_AMP_DECAY)
        if i > n - fade:                      # tiny fade so the tail never clicks
            env *= (n - i) / fade
        out.append(KICK_GAIN * math.tanh(KICK_DRIVE * env * math.sin(phase)))
    return out


def _render_bass_note(freq, sample_rate, beat):
    """One gated square-wave sub-bass note lasting BASS_GATE of a beat."""
    n = max(1, int(BASS_GATE * beat))
    attack = max(1, int(BASS_ATTACK * sample_rate))
    release = max(1, int(BASS_RELEASE * sample_rate))
    alpha = 1.0 - math.exp(-TWO_PI * BASS_LOWPASS_HZ / sample_rate)
    y = 0.0
    out = array("d")
    for i in range(n):
        square = 1.0 if math.sin(TWO_PI * freq * i / sample_rate) >= 0.0 else -1.0
        if i < attack:
            env = i / attack
        elif i > n - release:
            env = (n - i) / release
        else:
            env = 1.0
        y += alpha * (square * env - y)      # one-pole low-pass
        out.append(BASS_GAIN * y)
    return out


def _sidechain_envelope(beat, sample_rate, gain):
    """Gain curve for one beat: slam to the floor on the kick, swell back."""
    hold = int(SIDECHAIN_HOLD * sample_rate)
    release = max(1, int(SIDECHAIN_RELEASE * sample_rate))
    env = array("d")
    for i in range(beat):
        if i < hold:
            g = SIDECHAIN_FLOOR
        elif i < hold + release:
            x = (i - hold) / release
            g = SIDECHAIN_FLOOR + (1.0 - SIDECHAIN_FLOOR) * (1.0 - (1.0 - x) ** 2)
        else:
            g = 1.0
        env.append(g * gain)
    return env


# ---------------------------------------------------------------------------
# 1. Backing track
# ---------------------------------------------------------------------------

def generate_tech_house_backing(duration_secs, bpm=126, sample_rate=44100):
    """
    Build a tech-house loop of `duration_secs` seconds at `bpm`.

    * Kick drum (sine sliding 150Hz -> 40Hz) on every beat.
    * Square-wave sub-bass note on every off-beat (the "and" of each beat).

    Returns an array('d') of mono float samples in [-1.0, 1.0].  It behaves
    like a list of floats; wrap it with samples_to_wav_bytes() to get a WAV.
    """
    if duration_secs <= 0:
        raise ValueError("duration_secs must be positive")

    beat = _samples_per_beat(bpm, sample_rate)
    pattern_beats = len(BASS_PATTERN_MIDI)
    template = array("d", bytes(8 * beat * pattern_beats))     # all zeros

    kick = _render_kick(sample_rate)
    notes = {}
    for b in range(pattern_beats):
        start = b * beat
        _add_into(template, kick, start)                         # on the beat
        midi = BASS_PATTERN_MIDI[b]
        if midi not in notes:
            notes[midi] = _render_bass_note(_midi_hz(midi), sample_rate, beat)
        _add_into(template, notes[midi], start + beat // 2)      # off-beat

    total = int(round(duration_secs * sample_rate))
    return _tile(template, max(1, total))


def render_backing_wav(duration_secs, bpm=126, sample_rate=44100):
    """Convenience: backing track straight to WAV bytes."""
    return samples_to_wav_bytes(
        generate_tech_house_backing(duration_secs, bpm, sample_rate), sample_rate
    )


# ---------------------------------------------------------------------------
# WAV in / out
# ---------------------------------------------------------------------------

def _decode_wav_ints(wav_bytes):
    """WAV bytes -> (array of ints, channels, full_scale, sample_rate)."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            channels = w.getnchannels()
            width = w.getsampwidth()
            rate = w.getframerate()
            raw = w.readframes(w.getnframes())
    except (wave.Error, EOFError, struct.error) as exc:
        raise ValueError(
            "Could not read that WAV file (%s). Export the audio as a standard "
            "PCM WAV (16-bit or 24-bit) and try again." % exc
        ) from None

    frame_bytes = width * channels
    count = (len(raw) // frame_bytes) * channels          # total samples, all channels
    raw = raw[: count * width]

    if width == 1:                                         # 8-bit is unsigned
        ints = array("d", map(operator.sub, _int_array("B", raw), repeat(128)))
        full_scale = 128.0
    elif width == 2:
        ints = _int_array("h", raw)
        full_scale = 32768.0
    elif width == 3:                                       # 24-bit: pad to 32-bit
        buf = bytearray(count * 4)
        buf[1::4] = raw[0::3]
        buf[2::4] = raw[1::3]
        buf[3::4] = raw[2::3]
        ints = _int_array("i", bytes(buf))
        full_scale = 2147483648.0
    elif width == 4:
        ints = _int_array("i", raw)
        full_scale = 2147483648.0
    else:
        raise ValueError("Unsupported WAV sample width: %d bytes" % width)
    return ints, channels, full_scale, rate


def decode_wav(wav_bytes, mono=True):
    """
    WAV bytes -> (samples, sample_rate).

    mono=True  : one array('d') of floats in [-1, 1] (channels folded down).
    mono=False : a list with one array('d') per channel (1 = mono, 2 = L/R).
    Accepts 8/16/24/32-bit PCM at any sample rate and channel count.
    """
    ints, channels, full_scale, rate = _decode_wav_ints(wav_bytes)
    if mono:
        summed = ints[0::channels] if channels > 1 else ints
        for c in range(1, channels):
            summed = map(operator.add, summed, ints[c::channels])
        scale = 1.0 / (full_scale * channels)
        return array("d", map(operator.mul, summed, repeat(scale))), rate
    scale = 1.0 / full_scale
    chans = []
    for c in range(channels):
        chan = ints[c::channels] if channels > 1 else ints
        chans.append(array("d", map(operator.mul, chan, repeat(scale))))
    return chans, rate


def _decode_wav(wav_bytes):
    """Backwards-compatible alias: mono decode."""
    return decode_wav(wav_bytes, mono=True)


def samples_to_wav_bytes(samples, sample_rate=44100, stereo=True):
    """
    Master float samples to a 16-bit WAV file.

    `samples` is either one mono array (written to both channels when
    stereo=True) or a [left, right] pair of arrays for true stereo.
    Peak-normalises to MASTER_PEAK so nothing can clip.
    """
    if isinstance(samples, (list, tuple)) and samples and isinstance(samples[0], array):
        if len(samples) == 1:                       # [mono] -> mono
            samples = samples[0]
        elif len(samples) > 2:
            raise ValueError("samples_to_wav_bytes takes one array or [left, right]")
    if isinstance(samples, (list, tuple)):
        left, right = samples
        n = min(len(left), len(right))
        left, right = left[:n], right[:n]
        peak = max(max(map(abs, left), default=0.0), max(map(abs, right), default=0.0))
        gain = (MASTER_PEAK / peak) if peak > 0.0 else 1.0
        k = repeat(gain * 32767.0)
        frames = array("h", bytes(n * 4))
        frames[0::2] = array("h", map(int, map(operator.mul, left, k)))
        frames[1::2] = array("h", map(int, map(operator.mul, right, k)))
        channels = 2
    else:
        if not isinstance(samples, array):
            samples = array("d", samples)
        peak = max(map(abs, samples), default=0.0)
        gain = (MASTER_PEAK / peak) if peak > 0.0 else 1.0
        pcm = array("h", map(int, map(operator.mul, samples, repeat(gain * 32767.0))))
        if stereo:
            frames = array("h", bytes(len(pcm) * 4))
            frames[0::2] = pcm
            frames[1::2] = pcm
            channels = 2
        else:
            frames = pcm
            channels = 1

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(_native(frames))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Time-stretch / resample
# ---------------------------------------------------------------------------

def _resample(samples, step):
    """
    Linear-interpolation resample:  out[i] = samples[i * step].

    step < 1 stretches the audio longer, step > 1 squeezes it shorter.
    Works block by block on a plain-list window of the source so memory
    stays flat and every lookup is a C-level list index.
    """
    n_src = len(samples)
    if abs(step - 1.0) < 1e-12 or n_src < 2:
        return samples
    n_out = max(1, int((n_src - 1) / step))
    last = samples[-1]
    out = array("d")
    for a in range(0, n_out, _CHUNK):
        b = min(n_out, a + _CHUNK)
        lo = int(a * step)
        hi = min(n_src, int(b * step) + 3)
        win = samples[lo:hi].tolist() + [last, last]      # guard samples
        win1 = win[1:]                                     # pre-shifted copy
        pos = array("d", map(operator.add,
                             map(operator.mul, range(a, b), repeat(step)),
                             repeat(-lo)))
        i0 = array("q", map(int, pos))
        frac = array("d", map(operator.sub, pos, i0))
        s0 = list(map(win.__getitem__, i0))
        delta = map(operator.sub, map(win1.__getitem__, i0), s0)
        out.extend(map(operator.add, s0, map(operator.mul, delta, frac)))
    return out


def _grid_stretch_ratio(vocal_secs, bpm):
    """
    How much to stretch the vocal so it ends exactly on the BPM grid.

    Prefers landing on a whole number of bars when that is a subtle move
    (<= 6%), otherwise lands on a whole number of beats.  Ratio > 1 means
    the vocal gets longer.
    """
    beat = 60.0 / bpm
    bar = 4.0 * beat
    bars = max(1, int(round(vocal_secs / bar)))
    ratio = bars * bar / vocal_secs
    if abs(ratio - 1.0) > 0.06:
        beats = max(1, int(round(vocal_secs / beat)))
        ratio = beats * beat / vocal_secs
    return min(1.0 + MAX_GRID_STRETCH, max(1.0 - MAX_GRID_STRETCH, ratio))


def resample(samples, src_rate, dst_rate):
    """Convert `samples` from src_rate to dst_rate (linear interpolation)."""
    if int(src_rate) == int(dst_rate):
        return samples
    return _resample(samples, float(src_rate) / float(dst_rate))


def fit_length(samples, n):
    """Trim or zero-pad `samples` to exactly n samples."""
    if len(samples) == n:
        return samples
    if len(samples) > n:
        return samples[:n]
    return samples + array("d", bytes(8 * (n - len(samples))))


def butt_join(chunks, sample_rate, fade_secs=0.03):
    """
    Concatenate audio chunks end to end with a short fade at every seam.

    No overlap is used, so the total length is exactly the sum of the
    chunks and anything time-aligned to them (like a vocal) stays in sync.
    """
    chunks = [c for c in chunks if len(c)]
    if not chunks:
        return array("d")
    fade = max(1, int(fade_secs * sample_rate))
    out = array("d")
    last = len(chunks) - 1
    for idx, chunk in enumerate(chunks):
        chunk = array("d", chunk)
        f = min(fade, len(chunk) // 2)
        if f and idx > 0:                                     # fade in
            ramp = map(operator.mul, range(f), repeat(1.0 / f))
            chunk[:f] = array("d", map(operator.mul, chunk[:f], ramp))
        if f and idx < last:                                  # fade out
            ramp = map(operator.mul, range(f, 0, -1), repeat(1.0 / f))
            chunk[-f:] = array("d", map(operator.mul, chunk[-f:], ramp))
        out.extend(chunk)
    return out


def transient_duck_envelope(trigger, sample_rate, floor=DUCK_FLOOR,
                            hold=DUCK_HOLD, release=DUCK_RELEASE):
    """
    Build a per-sample gain curve from the hits in `trigger` (mono array).

    Every detected transient (kick, snare, stab) pulls the gain down to
    `floor`, holds it, then swells back to 1.0 over `release` seconds -
    a genre-agnostic sidechain that works with any instrumental.
    """
    n = len(trigger)
    block = DUCK_BLOCK
    n_blocks = (n + block - 1) // block
    block_secs = block / float(sample_rate)
    min_gap = max(1, int(DUCK_MIN_GAP / block_secs))

    # 1. coarse loudness per block (mean |x|), all C-level
    levels = []
    for b in range(n_blocks):
        seg = trigger[b * block:(b + 1) * block]
        levels.append(sum(map(abs, seg)) / len(seg))

    # 2. detect onsets against a running average, then shape the gain
    gains = []
    avg = 0.0                          # running average; 0 lets a hit on block 0 fire
    since = 1e9                        # seconds since the last hit
    last_hit = -min_gap
    for b, lvl in enumerate(levels):
        if (lvl > DUCK_MIN_LEVEL and lvl > DUCK_ONSET_RATIO * avg
                and b - last_hit >= min_gap):
            since = 0.0
            last_hit = b
        if since < hold:
            g = floor
        elif since < hold + release:
            x = (since - hold) / release
            g = floor + (1.0 - floor) * (1.0 - (1.0 - x) ** 2)
        else:
            g = 1.0
        gains.append(g)
        avg += 0.12 * (lvl - avg)
        since += block_secs
    gains.append(1.0)

    # 3. expand to per-sample with a linear ramp inside each block
    env = array("d")
    for b in range(n_blocks):
        g0, g1 = gains[b], gains[b + 1]
        size = min(block, n - b * block)
        env.extend(map(operator.add, repeat(g0, size),
                       map(operator.mul, range(size), repeat((g1 - g0) / block))))
    return env


def mix_stems(vocal_channels, inst_channels, sample_rate,
              vocal_gain=0.9, inst_gain=0.8, duck_floor=DUCK_FLOOR):
    """
    Lay isolated vocals over an instrumental.

    Both inputs are lists of channel arrays at `sample_rate` (1 or 2 each).
    The instrumental's transients duck the vocal (see transient_duck_envelope)
    so the two sit together instead of fighting.  Returns [left, right].
    """
    def stereo(chs):
        return [chs[0], chs[0]] if len(chs) == 1 else [chs[0], chs[1]]

    vox = stereo(vocal_channels)
    inst = stereo(inst_channels)
    n = max(len(vox[0]), len(inst[0]))
    vox = [fit_length(c, n) for c in vox]
    inst = [fit_length(c, n) for c in inst]

    # normalise each stem to a known peak before balancing
    vpeak = max(max(map(abs, vox[0]), default=0.0), max(map(abs, vox[1]), default=0.0)) or 1.0
    ipeak = max(max(map(abs, inst[0]), default=0.0), max(map(abs, inst[1]), default=0.0)) or 1.0

    mono_inst = array("d", map(operator.mul, map(operator.add, inst[0], inst[1]), repeat(0.5 / ipeak)))
    env = transient_duck_envelope(mono_inst, sample_rate, floor=duck_floor)
    env = array("d", map(operator.mul, env, repeat(vocal_gain / vpeak)))

    out = []
    for c in range(2):
        ducked = map(operator.mul, vox[c], env)
        bed = map(operator.mul, inst[c], repeat(inst_gain / ipeak))
        out.append(array("d", map(operator.add, ducked, bed)))
    return out


# ---------------------------------------------------------------------------
# 2. Remix
# ---------------------------------------------------------------------------

def remix_track(vocal_wav_bytes, bpm=126, sample_rate=44100, source_bpm=None,
                intro_bars=2, outro_bars=2):
    """
    Turn an uploaded vocal WAV into a finished tech-house remix.

    1. Decode the WAV (8/16/24/32-bit PCM, any channel count, any rate).
    2. Stretch it onto the `bpm` grid.  If you know the vocal's original
       tempo pass `source_bpm` for an exact tempo match; otherwise the clip
       is stretched so it ends on a whole bar/beat of the grid.
    3. Sidechain: the vocal is multiplied by 0.25 on every beat and swells
       back before the next kick (heavy pumping, John Summit style).
    4. Mix with generate_tech_house_backing(), with `intro_bars` of beat
       before the vocal drops and `outro_bars` after it ends.
    5. Master to a peak-normalised 16-bit stereo WAV.

    Returns the WAV file as bytes - write it straight to disk or stream it.
    """
    vocal, src_rate = _decode_wav(vocal_wav_bytes)
    if len(vocal) == 0:
        raise ValueError("The WAV file contains no audio.")

    # --- stretch onto the grid -------------------------------------------
    vocal_secs = len(vocal) / float(src_rate)
    if source_bpm:
        stretch = float(source_bpm) / float(bpm)       # 120 -> 126: play 5% faster
    else:
        stretch = _grid_stretch_ratio(vocal_secs, bpm)
    step = (float(src_rate) / float(sample_rate)) / stretch
    vocal = _resample(vocal, step)

    # --- lay the vocal out on a whole number of bars ---------------------
    beat = _samples_per_beat(bpm, sample_rate)
    bar = 4 * beat
    intro = int(intro_bars) * bar
    body = ((len(vocal) + bar - 1) // bar) * bar
    total = intro + body + int(outro_bars) * bar

    timeline = array("d", bytes(8 * total))                 # zeros
    timeline[intro:intro + len(vocal)] = vocal

    # --- sidechain duck on every beat (x0.25 on the kick, swell back) ----
    peak = max(map(abs, vocal), default=0.0) or 1.0
    env = _sidechain_envelope(beat, sample_rate, VOCAL_GAIN / peak)
    ducked = map(operator.mul, timeline, _tile(env, total))

    # --- mix with the backing and master ---------------------------------
    backing = generate_tech_house_backing(total / float(sample_rate), bpm, sample_rate)
    mix = array("d", map(operator.add, ducked, backing))
    return samples_to_wav_bytes(mix, sample_rate)


# ---------------------------------------------------------------------------
# CLI demo:  python engine.py                -> demo_backing.wav (8 bars)
#            python engine.py vocal.wav      -> vocal_remix.wav
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        with open(path, "rb") as f:
            data = f.read()
        out_path = path.rsplit(".", 1)[0] + "_remix.wav"
        with open(out_path, "wb") as f:
            f.write(remix_track(data))
        print("wrote", out_path)
    else:
        with open("demo_backing.wav", "wb") as f:
            f.write(render_backing_wav(8 * 4 * 60.0 / 126))
        print("wrote demo_backing.wav")
