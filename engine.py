"""
engine.py - DJAYAIIII core audio engine
=======================================

A zero-dependency tech-house remix engine written with nothing but the
Python standard library (wave, math, struct, array, io, operator, itertools).
There is nothing to pip-install: if Python 3.8+ runs, this runs.

Speed: every per-sample operation is pushed down into C-level builtins
(array, map, operator, slicing, struct).  Instruments are rendered once as
short templates and tiled across the track, so a full-length remix renders
in a few seconds instead of freezing the process in a Python for-loop.

Public API
----------
generate_tech_house_backing(duration_secs, bpm=126, sample_rate=44100)
    -> array('d') of mono float samples in [-1.0, 1.0]
       Kick on every beat, square-wave sub-bass on every off-beat.

remix_track(vocal_wav_bytes, bpm=126, sample_rate=44100, source_bpm=None,
            intro_bars=2, outro_bars=2)
    -> bytes of a finished 16-bit stereo WAV file.
       Reads the uploaded WAV, stretches it onto the BPM grid, sidechains
       it against the kick, mixes it with the backing and masters it.

samples_to_wav_bytes(samples, sample_rate=44100, stereo=True)
    -> bytes   float samples -> normalised 16-bit WAV file.

render_backing_wav(duration_secs, bpm=126, sample_rate=44100)
    -> bytes   quick preview of just the backing track.
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

def _decode_wav(wav_bytes):
    """WAV bytes -> (array('d') of mono floats in [-1, 1], sample_rate)."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            channels = w.getnchannels()
            width = w.getsampwidth()
            rate = w.getframerate()
            raw = w.readframes(w.getnframes())
    except (wave.Error, EOFError, struct.error) as exc:
        raise ValueError(
            "Could not read that WAV file (%s). Export the vocal as a standard "
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

    if channels > 1:                                       # fold down to mono
        mono = ints[0::channels]
        for c in range(1, channels):
            mono = map(operator.add, mono, ints[c::channels])
        ints = mono

    scale = 1.0 / (full_scale * channels)
    return array("d", map(operator.mul, ints, repeat(scale))), rate


def samples_to_wav_bytes(samples, sample_rate=44100, stereo=True):
    """
    Master float samples to a 16-bit WAV file.

    Peak-normalises to MASTER_PEAK (so nothing can clip) and writes a
    stereo file by default, which every player and DAW accepts.
    """
    if not isinstance(samples, array):
        samples = array("d", samples)
    peak = max(map(abs, samples), default=0.0)
    gain = (MASTER_PEAK / peak) if peak > 0.0 else 1.0
    pcm = array("h", map(int, map(operator.mul, samples, repeat(gain * 32767.0))))

    if stereo:
        frames = array("h", bytes(len(pcm) * 4))
        frames[0::2] = pcm
        frames[1::2] = pcm
    else:
        frames = pcm

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2 if stereo else 1)
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
