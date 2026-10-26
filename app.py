"""
app.py - DJAYAIIII universal AI remix server
============================================

Upload ANY song (MP3 or WAV) + describe ANY remix style in plain text, and
get back a mastered WAV where the song's isolated vocals/melody ride over a
brand-new AI-generated instrumental that matches the description.

Zero dependencies: the web server, the multipart upload parser, the HTTP
client for the generative API and all the audio maths (engine.py) use only
the Python standard library.

    python app.py            ->  http://localhost:8000

Pipeline (cloud mode, when an API key is configured)
----------------------------------------------------
  1. The song is uploaded to the provider's file store over HTTPS.
  2. A stem-separation model (Demucs) isolates the vocals.
  3. The melodic content is cut into <= 30 s windows and each window is sent
     to a melody-conditioned music generator (MusicGen) together with the
     user's style prompt, so the new instrumental follows the song's melody.
  4. The generated windows are joined, the isolated vocals are laid on top
     with transient ducking, and the result is mastered to a stereo WAV.
  5. The browser polls the job queue and auto-serves the download when done.

Without an API key the app falls back to the offline tech-house engine
(WAV uploads only) so the interface stays usable.
"""

import base64
import io
import json
import math
import mimetypes
import operator
import os
import re
import secrets
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from array import array
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import engine

# ===========================================================================
#  API AUTHENTICATION
# ===========================================================================
#
#  RECOMMENDED: create a file called `.env` next to app.py (it is listed in
#  .gitignore, so it can never be committed) containing one line:
#
#      REPLICATE_API_TOKEN=r8_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#
#  or export it as an environment variable before starting:
#
#      export REPLICATE_API_TOKEN="r8_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
#      python app.py
#
#  You can also paste the key between the quotes below for a quick local
#  test, but then do NOT commit this file: anything committed to GitHub is
#  in the repo history forever, and a leaked key can run up your bill.
#
#  The key is only ever sent as an Authorization header over HTTPS to the
#  provider's API.  It is never written to disk, logged, or exposed to the
#  browser.
#
#  Run `python app.py --check` to verify the key and the model schemas
#  before paying for a render.
#
API_KEY = ""   # <-- paste your key here, e.g. "r8_..."  (or use .env / the env var)

# Which generative backend to use: "auto" picks the cloud provider when a
# key is present and falls back to the offline engine otherwise.
PROVIDER = os.environ.get("DJAY_PROVIDER", "auto")          # auto | replicate | local

# ===========================================================================
#  Generative model configuration (Replicate)
# ===========================================================================
# Model names are "owner/name".  The client fetches each model's live input
# schema at start of a job and only sends fields that version accepts, so
# minor schema differences between versions are handled automatically.
MUSIC_MODEL = os.environ.get("DJAY_MUSIC_MODEL", "meta/musicgen")
STEM_MODEL = os.environ.get("DJAY_STEM_MODEL", "ryan5453/demucs")

MUSICGEN_VARIANT = "stereo-melody-large"   # melody-conditioned, stereo output
WINDOW_SECS = 30                           # MusicGen's maximum clip length
GENERATION_SEED = 4242                     # same seed per window = consistent vibe
POLL_SECS = 3.0                            # how often to poll the cloud queue
JOB_TIMEOUT_SECS = 45 * 60                 # give up on a cloud render after this
MAX_UPLOAD_BYTES = 80 * 1024 * 1024

MASTER_RATE = 44100
RENDER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "renders")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))


def _load_dotenv():
    """Load KEY=value lines from a `.env` file next to app.py, if present."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except OSError:
        pass


_load_dotenv()


def api_key():
    """The configured key: hard-coded placeholder first, then env / .env."""
    return (API_KEY or os.environ.get("REPLICATE_API_TOKEN", "")).strip()


# ===========================================================================
#  Minimal Replicate HTTP client (urllib only)
# ===========================================================================

class ProviderError(RuntimeError):
    pass


class ReplicateClient:
    """
    Thin wrapper over https://api.replicate.com/v1 using only urllib.

    Covers exactly what the remix pipeline needs: upload a file, look up a
    model's latest version + input schema, create a prediction, poll it,
    and download its output.
    """

    BASE = "https://api.replicate.com/v1"

    def __init__(self, token):
        if not token:
            raise ProviderError("No API key configured.")
        self._token = token
        self._models = {}

    # -- low level ---------------------------------------------------------
    def _request(self, method, url, body=None, content_type=None, retries=3):
        if not url.startswith("http"):
            url = self.BASE + url
        headers = {"Authorization": "Bearer " + self._token,
                   "User-Agent": "djayaiiii/1.0"}
        if content_type:
            headers["Content-Type"] = content_type
        last = None
        for attempt in range(retries):
            req = urllib.request.Request(url, data=body, method=method, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return resp.read(), resp.headers.get("Content-Type", "")
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:500]
                if exc.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    last = exc
                    continue
                raise ProviderError("API %s %s -> HTTP %d: %s" % (method, url, exc.code, detail))
            except urllib.error.URLError as exc:
                last = exc
                time.sleep(2 ** attempt)
        raise ProviderError("API unreachable: %s" % last)

    def _json(self, method, path, payload=None):
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        data, _ = self._request(method, path, body, "application/json" if body else None)
        return json.loads(data.decode("utf-8"))

    # -- files -------------------------------------------------------------
    def upload_file(self, data, filename, content_type):
        """Upload bytes to the Files API and return a URL usable as model input."""
        boundary = "----djay" + secrets.token_hex(12)
        parts = [
            ("--%s\r\nContent-Disposition: form-data; name=\"content\"; filename=\"%s\"\r\n"
             "Content-Type: %s\r\n\r\n" % (boundary, filename, content_type)).encode(),
            data,
            ("\r\n--%s--\r\n" % boundary).encode(),
        ]
        try:
            raw, _ = self._request("POST", "/files", b"".join(parts),
                                   "multipart/form-data; boundary=" + boundary)
            info = json.loads(raw.decode("utf-8"))
            url = (info.get("urls") or {}).get("get")
            if url:
                return url
        except ProviderError as exc:
            if len(data) > 8 * 1024 * 1024:
                raise
            sys.stderr.write("files API unavailable (%s); using data URI\n" % exc)
        # Fallback for small files: inline data URI.
        return "data:%s;base64,%s" % (content_type, base64.b64encode(data).decode("ascii"))

    # -- models / predictions ----------------------------------------------
    def model_info(self, model):
        """-> (version_id, {input_field: schema}) for the model's latest version."""
        if model not in self._models:
            info = self._json("GET", "/models/" + model)
            version = info.get("latest_version") or {}
            schema = (((version.get("openapi_schema") or {}).get("components") or {})
                      .get("schemas") or {}).get("Input") or {}
            props = schema.get("properties") or {}
            # resolve enum references so callers can inspect allowed values
            comps = ((version.get("openapi_schema") or {}).get("components") or {}).get("schemas") or {}
            for name, prop in list(props.items()):
                for sub in prop.get("allOf", []):
                    ref = sub.get("$ref", "")
                    target = comps.get(ref.rsplit("/", 1)[-1], {})
                    if "enum" in target:
                        prop["enum"] = target["enum"]
            if not version.get("id"):
                raise ProviderError("Model %s has no published version" % model)
            self._models[model] = (version["id"], props)
        return self._models[model]

    def build_input(self, model, desired):
        """
        Keep only the fields this model version actually declares.
        `desired` maps field -> value; a value may also be a list of
        alternative field names, e.g. {("output_format", "format"): "wav"}.
        """
        _, props = self.model_info(model)
        out = {}
        for key, value in desired.items():
            names = key if isinstance(key, tuple) else (key,)
            for name in names:
                if name in props:
                    enum = props[name].get("enum")
                    if enum and value not in enum:
                        break               # this version doesn't offer that value
                    out[name] = value
                    break
        return out

    def audio_field(self, model):
        """Name of the model's audio-file input."""
        _, props = self.model_info(model)
        for name in ("input_audio", "audio", "melody", "audio_file", "input"):
            if name in props:
                return name
        for name, prop in props.items():
            if prop.get("format") == "uri":
                return name
        raise ProviderError("Model %s has no file input" % model)

    def create_prediction(self, model, inputs):
        version, _ = self.model_info(model)
        return self._json("POST", "/predictions", {"version": version, "input": inputs})

    def get_prediction(self, prediction):
        url = (prediction.get("urls") or {}).get("get") or ("/predictions/" + prediction["id"])
        return self._json("GET", url)

    def wait(self, prediction, on_tick=None, timeout=JOB_TIMEOUT_SECS):
        """Poll until the prediction finishes. Returns the final prediction."""
        start = time.time()
        while True:
            status = prediction.get("status")
            if status == "succeeded":
                return prediction
            if status in ("failed", "canceled"):
                raise ProviderError("Cloud render %s: %s" % (status, prediction.get("error") or "no details"))
            if time.time() - start > timeout:
                raise ProviderError("Cloud render timed out after %d s" % timeout)
            if on_tick:
                on_tick(status or "queued")
            time.sleep(POLL_SECS)
            prediction = self.get_prediction(prediction)

    def download(self, url):
        if url.startswith("data:"):
            return base64.b64decode(url.split(",", 1)[1])
        req = urllib.request.Request(url, headers={"User-Agent": "djayaiiii/1.0"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            return resp.read()


def _collect_uris(output):
    """Flatten a prediction output into [(label, url), ...]."""
    found = []

    def walk(node, label):
        if isinstance(node, str) and (node.startswith("http") or node.startswith("data:")):
            found.append((label, node))
        elif isinstance(node, dict):
            for k, v in node.items():
                walk(v, k if label in ("", "output", "stems") else label + "." + k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, label or str(i))

    walk(output, "")
    return found


def stem_inputs():
    """Desired inputs for the stem-separation model (filtered per version)."""
    return {
        ("output_format", "format"): "wav",
        "model": "htdemucs",
        "shifts": 1,
    }


def music_inputs(prompt, duration):
    """Desired inputs for the music-generation model (filtered per version)."""
    return {
        "prompt": prompt,
        "model_version": MUSICGEN_VARIANT,
        "duration": duration,
        "continuation": False,
        "output_format": "wav",
        "normalization_strategy": "loudness",
        "seed": GENERATION_SEED,
    }


# ===========================================================================
#  Providers
# ===========================================================================

class GenerativeProvider:
    """Interface every backend implements."""
    name = "base"
    accepts = (".wav",)

    def render(self, song_bytes, filename, style, progress):
        """-> mastered WAV bytes.  `progress(stage, message, fraction)`."""
        raise NotImplementedError


class ReplicateProvider(GenerativeProvider):
    """
    Demucs (stem isolation) + MusicGen-melody (instrumental generation) on
    Replicate, stitched and mastered locally by engine.py.

    To use a different generative service (a Suno/Udio-style API, a
    self-hosted MusicGen, ...) subclass GenerativeProvider and plug it into
    make_provider() - the rest of the app does not care which one it is.
    """
    name = "replicate"
    accepts = (".wav", ".mp3", ".flac", ".m4a", ".ogg")

    def __init__(self, token):
        self.client = ReplicateClient(token)

    # -- step 1 + 2: upload and isolate the vocals --------------------------
    def _separate(self, song_bytes, filename, progress):
        c = self.client
        progress("upload", "Uploading song to the cloud engine", 0.03)
        ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        song_url = c.upload_file(song_bytes, filename, ctype)

        progress("stems", "Isolating vocals and melody (stem separation)", 0.08)
        inputs = c.build_input(STEM_MODEL, stem_inputs())
        inputs[c.audio_field(STEM_MODEL)] = song_url
        pred = c.create_prediction(STEM_MODEL, inputs)
        pred = c.wait(pred, lambda s: progress("stems", "Stem separation: %s" % s, 0.12))

        uris = _collect_uris(pred.get("output"))
        if not uris:
            raise ProviderError("Stem separation returned no audio")

        def pick(*words):
            for label, url in uris:
                tag = (label + " " + url.rsplit("/", 1)[-1]).lower()
                if any(w in tag for w in words):
                    return url
            return None

        vocals_url = pick("vocal")
        if not vocals_url:
            raise ProviderError("Could not find a vocals stem in the output: %s" % [u[0] for u in uris])
        others = [u for l, u in uris if u != vocals_url and
                  any(w in (l + u).lower() for w in ("other", "no_vocals", "bass", "guitar", "piano"))]

        progress("stems", "Downloading stems", 0.2)
        vocal_wav = c.download(vocals_url)
        vocals, v_rate = engine.decode_wav(vocal_wav, mono=False)
        # Melody reference for the generator = vocals + the melodic stems.
        melody, m_rate = engine.decode_wav(vocal_wav, mono=True)
        for url in others[:3]:
            chan, r = engine.decode_wav(c.download(url), mono=True)
            chan = engine.resample(chan, r, m_rate)
            n = max(len(chan), len(melody))
            melody = array("d", map(operator.add, engine.fit_length(melody, n), engine.fit_length(chan, n)))
        return vocals, v_rate, melody, m_rate

    # -- step 3: generate the new instrumental window by window -------------
    def _generate(self, melody, m_rate, style, progress):
        c = self.client
        total_secs = len(melody) / float(m_rate)
        n_win = max(1, int(math.ceil(total_secs / WINDOW_SECS)))
        win_len = len(melody) / float(n_win)                 # samples, equal windows
        win_secs = win_len / m_rate
        prompt = ("%s. Instrumental only, no vocals, follows the melody of the "
                  "reference audio, full arrangement, professional studio mix" % style.strip().rstrip("."))

        preds = []
        for i in range(n_win):
            a, b = int(round(i * win_len)), int(round((i + 1) * win_len))
            progress("generate", "Uploading melody section %d/%d" % (i + 1, n_win), 0.22 + 0.1 * i / n_win)
            ref_wav = engine.samples_to_wav_bytes(melody[a:b], m_rate, stereo=False)
            ref_url = c.upload_file(ref_wav, "melody_%02d.wav" % i, "audio/wav")
            inputs = c.build_input(MUSIC_MODEL, music_inputs(prompt, int(min(WINDOW_SECS, math.ceil(win_secs)))))
            inputs[c.audio_field(MUSIC_MODEL)] = ref_url
            preds.append((c.create_prediction(MUSIC_MODEL, inputs), b - a))

        chunks = []
        for i, (pred, n_samples) in enumerate(preds):
            progress("generate", "Generating section %d/%d" % (i + 1, n_win), 0.35 + 0.45 * i / n_win)
            pred = c.wait(pred, lambda s, i=i: progress(
                "generate", "Section %d/%d: %s" % (i + 1, n_win, s), 0.35 + 0.45 * (i + 0.5) / n_win))
            uris = _collect_uris(pred.get("output"))
            if not uris:
                raise ProviderError("Music generation returned no audio")
            chans, rate = engine.decode_wav(c.download(uris[0][1]), mono=False)
            target = int(round(n_samples * MASTER_RATE / float(m_rate)))
            chans = [engine.fit_length(engine.resample(ch, rate, MASTER_RATE), target) for ch in chans]
            chunks.append(chans)
        return chunks

    # -- full pipeline ------------------------------------------------------
    def render(self, song_bytes, filename, style, progress):
        vocals, v_rate, melody, m_rate = self._separate(song_bytes, filename, progress)
        chunks = self._generate(melody, m_rate, style, progress)

        progress("mix", "Stitching sections and laying the vocals on top", 0.85)
        n_ch = max(len(ch) for ch in chunks)
        inst = []
        for ci in range(n_ch):
            inst.append(engine.butt_join([ch[min(ci, len(ch) - 1)] for ch in chunks], MASTER_RATE))
        vocals = [engine.resample(ch, v_rate, MASTER_RATE) for ch in vocals]
        mixed = engine.mix_stems(vocals, inst, MASTER_RATE)

        progress("master", "Mastering", 0.95)
        return engine.samples_to_wav_bytes(mixed, MASTER_RATE)


class LocalProvider(GenerativeProvider):
    """Offline fallback: engine.py's synthesised tech-house remix (WAV only)."""
    name = "local"
    accepts = (".wav",)

    def render(self, song_bytes, filename, style, progress):
        progress("local", "No API key configured: rendering with the offline tech-house engine", 0.1)
        match = re.search(r"(\d{2,3})\s*bpm", style, re.IGNORECASE)
        bpm = int(match.group(1)) if match else 126
        bpm = min(200, max(60, bpm))
        progress("local", "Rendering at %d BPM" % bpm, 0.4)
        return engine.remix_track(song_bytes, bpm=bpm)


def make_provider():
    key = api_key()
    if PROVIDER == "local" or (PROVIDER == "auto" and not key):
        return LocalProvider()
    return ReplicateProvider(key)


# ===========================================================================
#  Job queue (background threads, polled by the browser)
# ===========================================================================

class Job:
    def __init__(self, filename, style):
        self.id = secrets.token_urlsafe(12)
        self.filename = filename
        self.style = style
        self.status = "queued"          # queued | running | done | error
        self.stage = "queued"
        self.message = "Waiting in queue"
        self.progress = 0.0
        self.log = []
        self.error = None
        self.output_path = None
        self.created = time.time()
        self.lock = threading.Lock()

    def update(self, stage, message, fraction):
        with self.lock:
            self.stage = stage
            self.message = message
            self.progress = max(self.progress, min(1.0, fraction))
            if not self.log or self.log[-1] != message:
                self.log.append(message)
                self.log = self.log[-40:]

    def to_json(self):
        with self.lock:
            return {
                "id": self.id, "status": self.status, "stage": self.stage,
                "message": self.message, "progress": round(self.progress, 3),
                "log": list(self.log), "error": self.error,
                "download_url": "/api/jobs/%s/download" % self.id if self.status == "done" else None,
                "audio_url": "/api/jobs/%s/audio" % self.id if self.status == "done" else None,
            }


class JobQueue:
    def __init__(self, workers=2):
        self.jobs = {}
        self.lock = threading.Lock()
        self.sem = threading.Semaphore(workers)
        os.makedirs(RENDER_DIR, exist_ok=True)

    def submit(self, song_bytes, filename, style):
        job = Job(filename, style)
        with self.lock:
            self.jobs[job.id] = job
        threading.Thread(target=self._run, args=(job, song_bytes), daemon=True).start()
        return job

    def get(self, job_id):
        with self.lock:
            return self.jobs.get(job_id)

    def _run(self, job, song_bytes):
        with self.sem:
            job.status = "running"
            try:
                provider = make_provider()
                ext = os.path.splitext(job.filename)[1].lower()
                if ext not in provider.accepts:
                    raise ProviderError(
                        "%s files need the cloud engine (set your API key); the offline "
                        "engine accepts %s" % (ext or "those", ", ".join(provider.accepts)))
                wav = provider.render(song_bytes, job.filename, job.style, job.update)
                path = os.path.join(RENDER_DIR, job.id + ".wav")
                with open(path, "wb") as f:
                    f.write(wav)
                job.output_path = path
                job.update("done", "Master ready - %.1f MB" % (len(wav) / 1e6), 1.0)
                job.status = "done"
            except Exception as exc:                       # surface every failure to the UI
                traceback.print_exc()
                job.error = str(exc)
                job.update("error", "Failed: %s" % exc, job.progress)
                job.status = "error"


QUEUE = JobQueue()


# ===========================================================================
#  HTTP layer
# ===========================================================================

def parse_multipart(body, content_type):
    """Tiny multipart/form-data parser -> {field: (filename, bytes)}."""
    match = re.search(r'boundary="?([^";]+)"?', content_type or "")
    if not match:
        raise ValueError("missing multipart boundary")
    boundary = b"--" + match.group(1).encode()
    fields = {}
    for part in body.split(boundary)[1:]:
        if part.startswith(b"--"):
            break
        part = part[2:] if part.startswith(b"\r\n") else part
        head, sep, data = part.partition(b"\r\n\r\n")
        if not sep:
            continue
        if data.endswith(b"\r\n"):
            data = data[:-2]
        disp = ""
        for line in head.decode("utf-8", "replace").split("\r\n"):
            if line.lower().startswith("content-disposition"):
                disp = line
        name = re.search(r'name="([^"]*)"', disp)
        fname = re.search(r'filename="([^"]*)"', disp)
        if name:
            fields[name.group(1)] = (fname.group(1) if fname else None, data)
    return fields


class Handler(BaseHTTPRequestHandler):
    server_version = "djayaiiii/1.0"

    def log_message(self, fmt, *args):          # quieter console
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -- helpers -----------------------------------------------------------
    def _send(self, code, body, ctype="text/html; charset=utf-8", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, payload):
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    # -- routes ------------------------------------------------------------
    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/":
            return self._send(200, INDEX_HTML.encode("utf-8"))
        if path == "/api/status":
            provider = make_provider()
            return self._json(200, {
                "provider": provider.name,
                "cloud": provider.name != "local",
                "accepts": list(provider.accepts),
                "music_model": MUSIC_MODEL, "stem_model": STEM_MODEL,
            })
        m = re.match(r"^/api/jobs/([A-Za-z0-9_-]+)(?:/(download|audio))?$", path)
        if m:
            job = QUEUE.get(m.group(1))
            if not job:
                return self._json(404, {"error": "unknown job"})
            if not m.group(2):
                return self._json(200, job.to_json())
            if job.status != "done" or not job.output_path:
                return self._json(409, {"error": "not ready"})
            with open(job.output_path, "rb") as f:
                data = f.read()
            extra = {"Accept-Ranges": "bytes"}
            if m.group(2) == "download":
                base = re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.splitext(job.filename)[0])[:40] or "track"
                extra["Content-Disposition"] = 'attachment; filename="%s_remix.wav"' % base
            return self._send(200, data, "audio/wav", extra)
        self._json(404, {"error": "not found"})

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if path != "/api/remix":
            return self._json(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return self._json(400, {"error": "empty upload"})
        if length > MAX_UPLOAD_BYTES:
            return self._json(413, {"error": "file too large (max %d MB)" % (MAX_UPLOAD_BYTES // 2**20)})
        body = self.rfile.read(length)
        try:
            fields = parse_multipart(body, self.headers.get("Content-Type"))
        except ValueError as exc:
            return self._json(400, {"error": str(exc)})
        song = fields.get("song")
        style = (fields.get("style", (None, b""))[1]).decode("utf-8", "replace").strip()
        if not song or not song[1]:
            return self._json(400, {"error": "please choose a song file"})
        if not style:
            return self._json(400, {"error": "please describe the remix style"})
        filename = os.path.basename(song[0] or "upload.wav")
        job = QUEUE.submit(song[1], filename, style[:500])
        self._json(202, {"job_id": job.id, "poll_url": "/api/jobs/" + job.id})


# ===========================================================================
#  Frontend
# ===========================================================================

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DJAYAIIII</title>
<style>
  :root { --bg:#0b0b10; --panel:#15151f; --line:#272738; --text:#f2f2f7; --muted:#9a9ab0;
          --accent:#7c5cff; --accent2:#ff5c8a; --ok:#39d98a; --err:#ff6b6b; }
  * { box-sizing:border-box; }
  body { margin:0; background:radial-gradient(1200px 600px at 50% -10%, #221a4a 0%, var(--bg) 60%);
         color:var(--text); font:16px/1.5 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; min-height:100vh; }
  main { max-width:720px; margin:0 auto; padding:48px 16px 80px; }
  h1 { font-size:40px; letter-spacing:-0.02em; margin:0 0 4px; }
  h1 span { background:linear-gradient(90deg,var(--accent),var(--accent2)); -webkit-background-clip:text; color:transparent; }
  .sub { color:var(--muted); margin:0 0 28px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:22px; }
  label { display:block; font-weight:600; margin:18px 0 8px; }
  label small { color:var(--muted); font-weight:400; margin-left:6px; }
  .drop { border:2px dashed var(--line); border-radius:12px; padding:22px; text-align:center; color:var(--muted);
          cursor:pointer; transition:.15s; }
  .drop.over, .drop:hover { border-color:var(--accent); color:var(--text); }
  .drop input { display:none; }
  .drop b { color:var(--text); }
  textarea { width:100%; min-height:96px; resize:vertical; background:#0f0f17; color:var(--text); border:1px solid var(--line);
             border-radius:12px; padding:14px; font:inherit; }
  textarea:focus { outline:none; border-color:var(--accent); }
  .chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
  .chip { font-size:13px; color:var(--muted); border:1px solid var(--line); border-radius:999px; padding:4px 10px; cursor:pointer; }
  .chip:hover { color:var(--text); border-color:var(--accent); }
  button.go { margin-top:22px; width:100%; padding:16px; border:0; border-radius:12px; font:inherit; font-weight:700; font-size:17px;
              color:#fff; background:linear-gradient(90deg,var(--accent),var(--accent2)); cursor:pointer; }
  button.go:disabled { opacity:.5; cursor:not-allowed; }
  .status { margin-top:22px; display:none; }
  .bar { height:10px; background:#0f0f17; border:1px solid var(--line); border-radius:999px; overflow:hidden; }
  .bar i { display:block; height:100%; width:0; background:linear-gradient(90deg,var(--accent),var(--accent2)); transition:width .6s; }
  .msg { margin:10px 0 0; color:var(--muted); font-size:14px; }
  .msg.err { color:var(--err); }
  .log { margin-top:10px; max-height:140px; overflow:auto; font:12px/1.6 ui-monospace, SFMono-Regular, Menlo, monospace; color:var(--muted);
         background:#0f0f17; border:1px solid var(--line); border-radius:10px; padding:10px 12px; }
  .result { display:none; margin-top:22px; }
  audio { width:100%; margin:8px 0 14px; }
  a.dl { display:inline-block; padding:12px 18px; border-radius:10px; background:var(--ok); color:#06160d; font-weight:700; text-decoration:none; }
  .mode { font-size:13px; color:var(--muted); margin-top:14px; }
  .mode b { color:var(--ok); }
  .mode b.off { color:#ffb347; }
</style>
</head>
<body>
<main>
  <h1>DJAY<span>AIIII</span></h1>
  <p class="sub">Any song. Any genre. Describe the remix, we build it.</p>

  <form class="card" id="form">
    <label for="song">Upload Song File <small>MP3 or WAV, any genre</small></label>
    <div class="drop" id="drop">
      <input type="file" id="song" name="song" accept=".mp3,.wav,audio/mpeg,audio/wav,audio/x-wav">
      <div id="dropText">Drop a track here or <b>click to choose</b></div>
    </div>

    <label for="style">Describe the Remix Style <small>free text</small></label>
    <textarea id="style" name="style" placeholder="e.g. John Summit Tech House with driving sub-bass"></textarea>
    <div class="chips" id="chips">
      <span class="chip">John Summit Tech House with driving sub-bass</span>
      <span class="chip">Fred again.. style emotional UK Garage</span>
      <span class="chip">Tiësto Mainstage Big Room Festival EDM</span>
      <span class="chip">Lo-fi hip hop, dusty drums, warm Rhodes</span>
      <span class="chip">Afro house, 122 BPM, organic percussion</span>
    </div>

    <button class="go" id="go" type="submit">Generate Remix</button>
    <div class="mode" id="mode"></div>

    <div class="status" id="status">
      <div class="bar"><i id="fill"></i></div>
      <p class="msg" id="msg"></p>
      <div class="log" id="log"></div>
    </div>

    <div class="result" id="result">
      <label>Your master is ready</label>
      <audio id="player" controls></audio>
      <a class="dl" id="dl" href="#" download>Download master WAV</a>
    </div>
  </form>
</main>

<script>
(function () {
  const $ = id => document.getElementById(id);
  const form = $('form'), song = $('song'), drop = $('drop'), dropText = $('dropText');
  const style = $('style'), go = $('go'), status = $('status'), fill = $('fill');
  const msg = $('msg'), log = $('log'), result = $('result'), player = $('player'), dl = $('dl');
  let accepts = ['.wav', '.mp3'];

  fetch('/api/status').then(r => r.json()).then(s => {
    accepts = s.accepts;
    $('mode').innerHTML = s.cloud
      ? 'Engine: <b>cloud generative AI</b> (' + s.stem_model + ' + ' + s.music_model + ')'
      : 'Engine: <b class="off">offline fallback</b> - add your API key in app.py to unlock any-genre AI remixes (WAV uploads only until then)';
  }).catch(() => {});

  const showFile = () => { if (song.files[0]) dropText.innerHTML = '<b>' + song.files[0].name + '</b> (' + (song.files[0].size / 1048576).toFixed(1) + ' MB)'; };
  drop.addEventListener('click', () => song.click());
  song.addEventListener('change', showFile);
  ['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', e => { if (e.dataTransfer.files[0]) { song.files = e.dataTransfer.files; showFile(); } });
  $('chips').addEventListener('click', e => { if (e.target.classList.contains('chip')) style.value = e.target.textContent; });

  function setStatus(text, frac, isErr) {
    status.style.display = 'block';
    msg.textContent = text; msg.className = 'msg' + (isErr ? ' err' : '');
    fill.style.width = Math.round(frac * 100) + '%';
  }

  async function poll(id) {
    const r = await fetch('/api/jobs/' + id);
    const j = await r.json();
    setStatus(j.message, j.progress, j.status === 'error');
    log.innerHTML = j.log.map(l => '&rsaquo; ' + l.replace(/</g, '&lt;')).join('<br>');
    log.scrollTop = log.scrollHeight;
    if (j.status === 'done') {
      player.src = j.audio_url; dl.href = j.download_url;
      result.style.display = 'block'; go.disabled = false;
      return;
    }
    if (j.status === 'error') { go.disabled = false; return; }
    setTimeout(() => poll(id), 2500);
  }

  form.addEventListener('submit', async e => {
    e.preventDefault();
    result.style.display = 'none';
    const f = song.files[0];
    if (!f) return setStatus('Choose a song file first.', 0, true);
    const ext = '.' + f.name.split('.').pop().toLowerCase();
    if (!accepts.includes(ext)) return setStatus('This engine accepts ' + accepts.join(', ') + ' files.', 0, true);
    if (!style.value.trim()) return setStatus('Describe the remix style first.', 0, true);
    go.disabled = true;
    setStatus('Uploading ' + f.name + '...', 0.01, false);
    const fd = new FormData(); fd.append('song', f); fd.append('style', style.value.trim());
    try {
      const r = await fetch('/api/remix', { method: 'POST', body: fd });
      const j = await r.json();
      if (!r.ok) { go.disabled = false; return setStatus(j.error || 'Upload failed', 0, true); }
      poll(j.job_id);
    } catch (err) { go.disabled = false; setStatus('Network error: ' + err, 0, true); }
  });
})();
</script>
</body>
</html>
"""


# ===========================================================================
#  Entry point
# ===========================================================================

def preflight():
    """`python app.py --check`: verify the key and model schemas, spend nothing."""
    key = api_key()
    if not key:
        print("No API key found. Put REPLICATE_API_TOKEN=... in a .env file next to app.py.")
        return 1
    print("key: %s...%s (%d chars)" % (key[:3], key[-4:], len(key)))
    try:
        c = ReplicateClient(key)
        acct = c._json("GET", "/account")
        print("account: %s (%s)" % (acct.get("username"), acct.get("type")))
        for label, model, desired in (
            ("stem model ", STEM_MODEL, stem_inputs()),
            ("music model", MUSIC_MODEL, music_inputs("<style prompt>", WINDOW_SECS)),
        ):
            version, props = c.model_info(model)
            sent = c.build_input(model, desired)
            sent[c.audio_field(model)] = "<audio file>"
            print("%s: %s @ %s" % (label, model, version[:12]))
            print("   schema fields: %s" % ", ".join(sorted(props)))
            print("   will send    : %s" % json.dumps(sent))
    except ProviderError as exc:
        print("FAILED: %s" % exc)
        return 1
    print("OK - ready to render.")
    return 0


def main():
    provider = make_provider()
    print("DJAYAIIII  ->  http://localhost:%d" % PORT)
    print("engine: %s%s" % (provider.name, "" if provider.name != "local"
                            else "  (no API key found - offline tech-house fallback, WAV only)"))
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(preflight())
    main()
