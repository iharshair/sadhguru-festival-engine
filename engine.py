#!/usr/bin/env python3
"""
Sadhguru Tiles - Festival Poster Engine v2
Free-tier only: GitHub Actions + Gemini free tier + Cloudflare Workers AI + WhatsApp Cloud API (+ optional Telegram).

Pipeline per festival:
  1. Gemini = art director  -> scene prompt + Devanagari title/subtitle (structured JSON)
  2. Cloudflare = N candidate scenes (FLUX.2 klein-4b + FLUX.1 schnell @ 8 steps)
  3. Gemini vision = judge  -> picks the best candidate (no text, good anatomy, clean wall)
  4. Pillow/NumPy = grade, light rays, adaptive scrim, real logo, metallic Devanagari title
  5. Delivery with retries; WhatsApp document -> template nudge on 24h-window errors -> Telegram backup
"""
import os, sys, io, json, glob, math, time, base64, random, textwrap, traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests
from PIL import (Image, ImageDraw, ImageFont, ImageFilter, ImageChops, features)

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
E = os.getenv
GEMINI_API_KEY = E("GEMINI_API_KEY", "")
CF_ACCOUNT_ID = E("CF_ACCOUNT_ID", "")
CF_API_TOKEN = E("CF_API_TOKEN", "")
WA_PHONE_NUMBER_ID = E("WA_PHONE_NUMBER_ID", "")
WA_ACCESS_TOKEN = E("WA_ACCESS_TOKEN", "")
CLIENT_WA_NUMBER = E("CLIENT_WA_NUMBER", "")
TG_TOKEN = E("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = E("TELEGRAM_CHAT_ID", "")

WA_TEMPLATE_NAME = E("WA_TEMPLATE_NAME", "hello_world")      # nudge template used if the 24h window is closed
WA_TEMPLATE_LANG = E("WA_TEMPLATE_LANG", "en_US")
POSTER_LANG = E("POSTER_LANG", "hindi").lower()               # "hindi" or "marathi" (can be overridden per festival)
CANDIDATE_PLAN = [c.strip() for c in E("CANDIDATES", "klein,schnell,klein,schnell").split(",") if c.strip()]
DRY_RUN = E("DRY_RUN", "").lower() in ("1", "true", "yes")

GEMINI_MODELS = [m for m in [E("GEMINI_MODEL"), "gemini-flash-latest", "gemini-2.5-flash", "gemini-2.5-flash-lite"] if m]

FESTIVAL_FILE = "data/festivals_2026_2027.json"
LOGO_PATH = E("LOGO_PATH", "assets/sadhguru_tiles_logo.png")
SENT_LOG = Path("data/sent_log.json")
OUT_DIR = Path("out")
FONT_DIR = Path("assets/fonts")
S = 1536  # final canvas (px). 1080 was fine, 1536 matches the ChatGPT references and looks sharper on phones.

GOOGLE_FONTS_RAW = "https://raw.githubusercontent.com/google/fonts/main/"
FONT_FILES = {
    "deva_title": ("YatraOne-Regular.ttf", ["ofl/yatraone/YatraOne-Regular.ttf"]),
    "deva_title_alt": ("RozhaOne-Regular.ttf", ["ofl/rozhaone/RozhaOne-Regular.ttf"]),
    "deva_body": ("TiroDevanagariMarathi-Regular.ttf", ["ofl/tirodevanagarimarathi/TiroDevanagariMarathi-Regular.ttf"]),
    "latin": ("Cinzel.ttf", ["ofl/cinzel/Cinzel%5Bwght%5D.ttf"]),
}
FONT_PATHS = {}
HAS_RAQM = features.check("raqm")

# Metallic palettes: (position 0..1 top->bottom, RGB)
THEMES = {
    "gold": [(0.00, (255, 246, 205)), (0.30, (240, 204, 115)), (0.55, (196, 146, 56)), (0.78, (246, 218, 138)), (1.00, (178, 128, 48))],
    "saffron": [(0.00, (255, 238, 190)), (0.30, (250, 190, 90)), (0.55, (222, 130, 40)), (0.78, (250, 205, 120)), (1.00, (190, 100, 30))],
    "rose_gold": [(0.00, (255, 236, 226)), (0.30, (236, 190, 165)), (0.55, (198, 138, 112)), (0.78, (244, 208, 186)), (1.00, (170, 112, 92))],
    "emerald_gold": [(0.00, (255, 246, 205)), (0.30, (236, 205, 120)), (0.55, (176, 150, 62)), (0.78, (240, 220, 140)), (1.00, (150, 122, 46))],
}


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------
# HTTP with exponential backoff
# ----------------------------------------------------------------------------
class ApiError(Exception):
    def __init__(self, status, body):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status, self.body = status, body


RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def http(method, url, retries=5, base=2.0, timeout=60, **kw):
    """requests wrapper: retries network errors and 429/5xx with exponential backoff + jitter."""
    last = None
    for attempt in range(1, retries + 1):
        delay = min(60.0, base ** attempt) + random.uniform(0, 1)
        try:
            r = requests.request(method, url, timeout=timeout, **kw)
            if r.status_code < 400:
                return r
            if r.status_code not in RETRY_STATUS:
                raise ApiError(r.status_code, r.text)
            last = ApiError(r.status_code, r.text)
            ra = r.headers.get("Retry-After", "")
            if ra.isdigit():
                delay = max(delay, float(ra))
        except (requests.ConnectionError, requests.Timeout) as e:
            last = e
        if attempt < retries:
            log(f"  retry {attempt}/{retries - 1} in {delay:.0f}s ({last})")
            time.sleep(delay)
    raise last if isinstance(last, ApiError) else RuntimeError(f"network failure: {last}")


# ----------------------------------------------------------------------------
# Calendar + idempotency
# ----------------------------------------------------------------------------
def get_festivals():
    ist = timezone(timedelta(hours=5, minutes=30))
    t = datetime.now(ist) + timedelta(days=1)
    target = t.strftime("%Y-%m-%d")
    override = E("TEST_DATE", "").strip()
    if override:
        target = override
    log(f"Target date: {target}")
    with open(FESTIVAL_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return target, [x for x in data.get(target.split("-")[0], []) if x.get("date") == target]


def slugify(s):
    return "".join(c if c.isalnum() else "_" for c in s.lower()).strip("_")


def load_sent():
    try:
        return set(json.loads(SENT_LOG.read_text()))
    except Exception:
        return set()


def mark_sent(key):
    s = load_sent()
    s.add(key)
    SENT_LOG.write_text(json.dumps(sorted(s), indent=1))


# ----------------------------------------------------------------------------
# Fonts
# ----------------------------------------------------------------------------
def _is_font(b):
    return b[:4] in (b"\x00\x01\x00\x00", b"OTTO", b"true", b"ttcf")


def ensure_fonts():
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    for role, (fname, paths) in FONT_FILES.items():
        dest = FONT_DIR / fname
        if dest.exists() and dest.stat().st_size > 10_000:
            FONT_PATHS[role] = str(dest)
            continue
        for p in paths:
            try:
                r = http("GET", GOOGLE_FONTS_RAW + p, retries=3, timeout=60)
                if _is_font(r.content[:4]):
                    dest.write_bytes(r.content)
                    FONT_PATHS[role] = str(dest)
                    log(f"  font downloaded: {fname}")
                    break
            except Exception as e:
                log(f"  font download failed {p}: {e}")
    sys_deva = sorted(glob.glob("/usr/share/fonts/**/*Devanagari*.ttf", recursive=True) +
                      glob.glob("/usr/share/fonts/**/*deva*.ttf", recursive=True))
    for role in ("deva_title", "deva_title_alt", "deva_body"):
        if role not in FONT_PATHS and sys_deva:
            FONT_PATHS[role] = sys_deva[0]
    if "latin" not in FONT_PATHS:
        for c in ("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
            if os.path.exists(c):
                FONT_PATHS["latin"] = c
                break
    log(f"Fonts: {FONT_PATHS} | raqm={HAS_RAQM}")


_font_cache = {}


def font(role, size):
    size = max(8, int(size))
    key = (role, size)
    if key in _font_cache:
        return _font_cache[key]
    path = FONT_PATHS.get(role)
    engine = ImageFont.Layout.RAQM if HAS_RAQM else ImageFont.Layout.BASIC
    try:
        f = ImageFont.truetype(path, size, layout_engine=engine)
        if role == "latin" and path and path.endswith("Cinzel.ttf"):
            try:
                f.set_variation_by_axes([700])
            except Exception:
                pass
    except Exception:
        f = ImageFont.load_default(size)
    _font_cache[key] = f
    return f


# ----------------------------------------------------------------------------
# Gemini: art director + judge
# ----------------------------------------------------------------------------
ART_SYSTEM = """You are the creative director of Sadhguru Tiles, a premium marble, granite and ceramic tile house in Maharashtra, India.
You make festival posters that look like Architectural Digest photography fused with authentic Indian festival heritage.
Return JSON only.

scene_prompt (goes to FLUX; FLUX has NO negative prompt, so describe ONLY what you want, positively):
- One paragraph, 90-120 words, editorial photography language: "ultra-realistic editorial photograph, 35mm lens, f/2.8, natural film grain, 8k, photorealistic".
- COMPOSITION (mandatory): square frame. The hero subject(s) sit in the lower {hero_side} 40% of the frame, sharp, anatomically correct, naturally posed.
  The opposite side and the whole upper area is one vast, smooth, perfectly blank polished bronze-brown marble feature wall with fine gold veining, empty like an unpainted canvas, softly lit, with a subtle arched niche outline in relief.
  The bottom 20% is glossy polished marble floor with warm reflections.
- LIGHT: warm golden-hour sunlight from the upper {light_side}, soft diagonal beams, dappled leaf shadows, glowing brass diyas.
- Use accurate festival props (brass lamps, marigold garlands, kalash, bells, brass thali, rangoli inlay in the marble floor, bow and arrows, lotus, tulsi, conch, etc.). Never depict deity faces or human faces; use symbolic objects and animals.
- The wall is a blank wall with no writing of any kind. Never mention text, letters, logos or signs as content.
Devanagari copy must be correct, standard spelling, no emojis, no brand name.
"""

ART_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "scene_prompt": {"type": "STRING"},
        "title_prefix_deva": {"type": "STRING"},
        "title_deva": {"type": "STRING"},
        "subtitle_deva_lines": {"type": "ARRAY", "items": {"type": "STRING"}},
        "subtitle_en": {"type": "STRING"},
        "theme": {"type": "STRING", "enum": list(THEMES.keys())},
    },
    "required": ["scene_prompt", "title_deva", "subtitle_deva_lines", "subtitle_en", "theme"],
}

JUDGE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "score": {"type": "NUMBER"},
        "has_text": {"type": "BOOLEAN"},
        "anatomy_ok": {"type": "BOOLEAN"},
        "text_zone_clear": {"type": "BOOLEAN"},
        "reason": {"type": "STRING"},
    },
    "required": ["score", "has_text", "anatomy_ok", "text_zone_clear"],
}


def gemini_json(parts, schema, system=None, temperature=0.9):
    if not GEMINI_API_KEY:
        raise RuntimeError("no GEMINI_API_KEY")
    last = None
    for model in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": temperature, "responseMimeType": "application/json", "responseSchema": schema},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        try:
            r = http("POST", url, headers={"x-goog-api-key": GEMINI_API_KEY}, json=payload, retries=3, timeout=60)
            txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(txt)
        except Exception as e:
            last = e
            log(f"  gemini model {model} failed: {e}")
    raise RuntimeError(f"all Gemini models failed: {last}")


def art_direct(fest, text_side):
    hero_side = "left" if text_side == "right" else "right"
    light_side = text_side  # light enters from the wall side so it rakes across the marble
    lang = (fest.get("poster_lang") or POSTER_LANG).lower()
    fallback_scene = (
        f"Ultra-realistic editorial photograph, 35mm lens, f/2.8, natural film grain, 8k. {fest['visual_prompt']}. "
        f"Hero subject in the lower {hero_side} of the frame. The rest is a vast, smooth, blank polished bronze-brown marble wall "
        f"with fine gold veining, softly lit by warm golden-hour sunlight from the upper {light_side}, dappled leaf shadows, "
        f"glossy marble floor with warm reflections, brass diyas glowing. A blank wall with no writing."
    )
    copy = {"scene_prompt": fallback_scene, "title_prefix_deva": "", "title_deva": "", "subtitle_deva_lines": [],
            "subtitle_en": fest.get("subtitle_en", ""), "theme": "gold"}
    try:
        msg = (f"Festival: {fest['festival_title_en']}\nDate: {fest['date']}\nVisual concept: {fest['visual_prompt']}\n"
               f"Existing Marathi greeting (for tone): {fest.get('greeting_marathi', '')}\n"
               f"Poster language for subtitle_deva_lines: {lang} (2-3 short lines, each at most 34 characters, a warm blessing about home, peace and prosperity).\n"
               f"title_prefix_deva: 'शुभ' if natural for this festival, else ''. title_deva: festival name only.\n"
               f"subtitle_en: at most 14 words.")
        out = gemini_json([{"text": msg}], ART_SCHEMA,
                          system=ART_SYSTEM.format(hero_side=hero_side, light_side=light_side))
        copy.update({k: v for k, v in out.items() if v})
        copy["theme"] = copy["theme"] if copy["theme"] in THEMES else "gold"
    except Exception as e:
        log(f"Art director fallback used: {e}")
    # Calendar JSON always wins when it provides verified copy
    for k in ("title_deva", "title_prefix_deva", "subtitle_deva_lines", "theme"):
        if fest.get(k):
            copy[k] = fest[k]
    copy["subtitle_deva_lines"] = [l for l in copy.get("subtitle_deva_lines", []) if l][:3]
    log(f"Scene prompt: {copy['scene_prompt'][:140]}...")
    return copy


def judge(img, fest):
    small = img.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    small.save(buf, "JPEG", quality=85)
    rubric = (
        f"You are a strict photo editor judging a candidate background for a luxury festival poster ({fest['festival_title_en']}). "
        "Score 0-10 for photorealism, premium marble/interior quality, correct festival props, warm elegant lighting. "
        "has_text: true if ANY letters/words/logos/signage are visible. anatomy_ok: false if animals/objects are deformed. "
        "text_zone_clear: true if there is a large clean, mostly empty wall area (a third of the frame or more) usable for typography."
    )
    parts = [{"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(buf.getvalue()).decode()}}, {"text": rubric}]
    return gemini_json(parts, JUDGE_SCHEMA, temperature=0.2)


# ----------------------------------------------------------------------------
# Cloudflare Workers AI
# ----------------------------------------------------------------------------
def _cf_image(r):
    if r.headers.get("content-type", "").startswith("image/"):
        data = r.content
    else:
        data = base64.b64decode(r.json()["result"]["image"])
    return Image.open(io.BytesIO(data)).convert("RGB")


def _cf_url(model):
    return f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{model}"


def gen_klein(prompt, size=1024):
    """FLUX.2 [klein] 4B (Apache-2.0, safe for commercial use). Multipart form, fixed 4 steps."""
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}"}
    seed = random.randint(1, 2**31 - 1)
    base = {"prompt": (None, prompt), "width": (None, str(size)), "height": (None, str(size))}
    try:
        r = http("POST", _cf_url("@cf/black-forest-labs/flux-2-klein-4b"), headers=headers,
                 files={**base, "seed": (None, str(seed))}, timeout=180)
    except ApiError as e:
        if e.status != 400:
            raise
        r = http("POST", _cf_url("@cf/black-forest-labs/flux-2-klein-4b"), headers=headers, files=base, timeout=180)
    return _cf_image(r)


def gen_schnell(prompt):
    """FLUX.1 [schnell] at the free-tier maximum of 8 steps."""
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}"}
    payload = {"prompt": prompt, "steps": 8, "seed": random.randint(1, 2**31 - 1)}
    try:
        r = http("POST", _cf_url("@cf/black-forest-labs/flux-1-schnell"), headers=headers, json=payload, timeout=180)
    except ApiError as e:
        if e.status != 400:
            raise
        payload.pop("seed")
        r = http("POST", _cf_url("@cf/black-forest-labs/flux-1-schnell"), headers=headers, json=payload, timeout=180)
    return _cf_image(r)


def make_candidates(prompt, tag):
    cands = []
    OUT_DIR.mkdir(exist_ok=True)
    for i, kind in enumerate(CANDIDATE_PLAN):
        try:
            img = gen_klein(prompt) if kind == "klein" else gen_schnell(prompt)
            img.convert("RGB").save(OUT_DIR / f"raw_{tag}_{i}_{kind}.jpg", quality=92)
            cands.append((kind, img))
            log(f"  candidate {i} ({kind}) ok {img.size}")
        except Exception as e:
            log(f"  candidate {i} ({kind}) failed: {e}")
    if not cands:
        raise RuntimeError("no image candidates could be generated")
    return cands


def pick_best(cands, fest):
    scored = []
    for kind, img in cands:
        try:
            j = judge(img, fest)
            ok = (not j.get("has_text")) and j.get("anatomy_ok", True)
            rank = float(j.get("score", 0)) + (1.5 if j.get("text_zone_clear") else 0) + (0 if ok else -20)
            scored.append((rank, kind, img))
            log(f"  judge {kind}: {j}")
        except Exception as e:
            log(f"  judge failed ({e}); keeping candidate unscored")
            scored.append((0.0, kind, img))
    scored.sort(key=lambda x: x[0], reverse=True)
    log(f"  winner: {scored[0][1]} (rank {scored[0][0]:.1f})")
    return scored[0][2]


# ----------------------------------------------------------------------------
# Image FX
# ----------------------------------------------------------------------------
def grade(img, side):
    a = np.asarray(img.convert("RGB"), np.float32) / 255.0
    a = np.clip(a * np.array([1.045, 1.0, 0.94], np.float32), 0, 1)
    s = a * a * (3 - 2 * a)
    a = a * 0.72 + s * 0.28                                   # gentle S-curve
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) / S
    r = np.sqrt(((xx - 0.5) / 0.78) ** 2 + ((yy - 0.5) / 0.78) ** 2)
    a *= (1 - 0.30 * np.clip(r, 0, 1.3) ** 2.2)[..., None]    # vignette
    # warm diagonal light shafts, strongest in the corner the light comes from
    cx = 1.0 if side == "right" else 0.0
    u = (xx * (0.8 if side == "right" else -0.8) + yy * 0.6)
    rng = np.random.default_rng(7)
    rays = np.zeros_like(u)
    for f in rng.uniform(9, 22, 5):
        rays += np.clip(np.sin(u * f + rng.uniform(0, 6.28)), 0, 1) ** 7 * rng.uniform(0.4, 1.0)
    fall = np.clip(1 - np.sqrt((xx - cx) ** 2 + yy ** 2) / 1.15, 0, 1) ** 1.6
    rays = (rays / max(rays.max(), 1e-6)) * fall
    rl = Image.fromarray((np.clip(rays, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(S * 0.004))
    rays = np.asarray(rl, np.float32) / 255.0
    layer = rays[..., None] * np.array([1.0, 0.82, 0.55], np.float32) * 0.26
    a = 1 - (1 - np.clip(a, 0, 1)) * (1 - layer)              # screen blend
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).convert("RGBA")


def zone_px(side):
    if side == "right":
        x0, x1 = 0.37, 0.95
    elif side == "left":
        x0, x1 = 0.05, 0.63
    else:
        x0, x1 = 0.17, 0.83
    return int(x0 * S), int(0.035 * S), int(x1 * S), int(0.86 * S)


def text_scrim(canvas, zone):
    x0, y0, x1, y1 = zone
    crop = canvas.crop(zone).convert("L")
    lum = float(np.asarray(crop, np.float32).mean())
    strength = float(np.clip((lum - 55) / 180 * 0.55 + 0.16, 0.16, 0.62))
    m = Image.new("L", (S, S), 0)
    ImageDraw.Draw(m).rounded_rectangle([x0, y0, x1, y1], radius=int(S * 0.05), fill=255)
    m = m.filter(ImageFilter.GaussianBlur(S * 0.055)).point(lambda v: int(v * strength))
    dark = Image.new("RGBA", (S, S), (16, 11, 7, 255))
    dark.putalpha(m)
    canvas.alpha_composite(dark)
    log(f"  zone luminance {lum:.0f} -> scrim {strength:.2f}")


def vgradient(y0, y1, stops):
    t = np.clip((np.arange(S, dtype=np.float32) - y0) / max(1, (y1 - y0)), 0, 1)
    pos = [p for p, _ in stops]
    chans = [np.interp(t, pos, [c[i] for _, c in stops]) for i in range(3)]
    row = np.stack(chans, axis=-1).astype(np.uint8)            # (S,3)
    return Image.fromarray(np.repeat(row[:, None, :], S, axis=1))


def shadow_glow(canvas, mask, glow=0.35, shadow=0.6, glow_rgb=(255, 205, 120)):
    if glow:
        g = mask.filter(ImageFilter.GaussianBlur(S * 0.016)).point(lambda v: int(v * glow))
        lay = Image.new("RGBA", (S, S), glow_rgb + (255,))
        lay.putalpha(g)
        canvas.alpha_composite(lay)
    if shadow:
        sh = ImageChops.offset(mask, 0, int(S * 0.006)).filter(ImageFilter.GaussianBlur(S * 0.007)).point(lambda v: int(v * shadow))
        lay = Image.new("RGBA", (S, S), (0, 0, 0, 255))
        lay.putalpha(sh)
        canvas.alpha_composite(lay)


def paint_metal(canvas, mask, y0, y1, stops, glow=0.35, shadow=0.6, bevel=True):
    shadow_glow(canvas, mask, glow, shadow)
    fill = vgradient(y0, y1, stops).convert("RGBA")
    fill.putalpha(mask)
    canvas.alpha_composite(fill)
    if bevel:
        k = max(2, int(S * 0.0022))
        hi = ImageChops.subtract(mask, ImageChops.offset(mask, k, k)).filter(ImageFilter.GaussianBlur(0.8)).point(lambda v: int(v * 0.85))
        lo = ImageChops.subtract(mask, ImageChops.offset(mask, -k, -k)).filter(ImageFilter.GaussianBlur(0.8)).point(lambda v: int(v * 0.5))
        for rgb, m in (((255, 252, 235), hi), ((70, 40, 12), lo)):
            lay = Image.new("RGBA", (S, S), rgb + (255,))
            lay.putalpha(m)
            canvas.alpha_composite(lay)


def measure(text, fnt):
    l, t, r, b = fnt.getbbox(text, anchor="ls")
    return l, t, r, b


def fit_font(role, text, start, max_w):
    size = start
    while size > 20:
        f = font(role, size)
        l, t, r, b = measure(text, f)
        if r - l <= max_w:
            return f
        size = int(size * 0.94)
    return font(role, size)


def draw_text_mask(text, fnt, cx, top):
    """Draw text centred on cx with its ink top at `top`. Returns (mask, bottom_y)."""
    l, t, r, b = measure(text, fnt)
    base = top - t
    m = Image.new("L", (S, S), 0)
    ImageDraw.Draw(m).text((cx - (l + r) / 2, base), text, font=fnt, fill=255, anchor="ls")
    return m, base + b


def lens(cx, cy, ang, length, width, n=20):
    a = math.radians(ang)
    ux, uy = math.cos(a), math.sin(a)
    vx, vy = -uy, ux
    top, bot = [], []
    for i in range(n + 1):
        t = i / n
        w = width * math.sin(math.pi * t) ** 0.85
        px, py = cx + ux * length * t, cy + uy * length * t
        top.append((px + vx * w, py + vy * w))
        bot.append((px - vx * w, py - vy * w))
    return top + bot[::-1]


def ornament_mask(cx, cy, half_w):
    m = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(m)
    L = S * 0.034
    for ang, ln in ((-90, 1.0), (-122, 0.8), (-58, 0.8), (-152, 0.55), (-28, 0.55)):
        pts = lens(cx, cy + L * 0.45, ang, L * ln, L * 0.2 * ln)
        d.line(pts + [pts[0]], fill=255, width=max(2, int(S * 0.0024)), joint="curve")
    gap = L * 1.5
    for sgn in (-1, 1):
        d.line([(cx + sgn * gap, cy), (cx + sgn * half_w, cy)], fill=255, width=max(2, int(S * 0.0022)))
        ex = cx + sgn * half_w
        r = S * 0.0045
        d.polygon([(ex, cy - r), (ex + r * 1.3, cy), (ex, cy + r), (ex - r * 1.3, cy)], fill=255)
    return m


def frame_mask():
    m = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(m)
    ins, ln = int(S * 0.028), int(S * 0.07)
    d.rectangle([ins, ins, S - ins, S - ins], outline=110, width=2)
    for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        x = ins if sx == 1 else S - ins
        y = ins if sy == 1 else S - ins
        d.line([(x, y), (x + sx * ln, y)], fill=255, width=4)
        d.line([(x, y), (x, y + sy * ln)], fill=255, width=4)
        r = S * 0.007
        px, py = x + sx * ln * 0.55, y + sy * ln * 0.55
        d.polygon([(px, py - r), (px + r, py), (px, py + r), (px - r, py)], fill=255)
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=255)
    return m


def tracked(text, fnt, cx, top, track):
    widths = [fnt.getlength(c) for c in text]
    total = sum(widths) + track * (len(text) - 1)
    x = cx - total / 2
    l, t, r, b = measure("H", fnt)
    m = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(m)
    for c, w in zip(text, widths):
        d.text((x, top - t), c, font=fnt, fill=255, anchor="ls")
        x += w + track
    return m


# ----------------------------------------------------------------------------
# Composition
# ----------------------------------------------------------------------------
def compose(art, fest, copy, out_path, side="right"):
    img = art.convert("RGB").resize((S, S), Image.Resampling.LANCZOS).filter(ImageFilter.UnsharpMask(1.4, 55, 3))
    canvas = grade(img, side)
    zone = zone_px(side)
    text_scrim(canvas, zone)
    x0, y0, x1, y1 = zone
    cx, zw = (x0 + x1) // 2, x1 - x0
    stops = THEMES.get(copy.get("theme", "gold"), THEMES["gold"])

    # bottom brand band
    band = np.zeros((S, S), np.uint8)
    h = int(S * 0.09)
    band[S - h:, :] = (np.linspace(0, 150, h)[:, None]).astype(np.uint8)
    b = Image.new("RGBA", (S, S), (12, 8, 5, 255))
    b.putalpha(Image.fromarray(band))
    canvas.alpha_composite(b)

    use_deva = bool(copy.get("title_deva")) and HAS_RAQM and "deva_title" in FONT_PATHS
    y = int(S * 0.055)

    # logo (real asset, with halo + shadow so it pops on any background)
    if os.path.exists(LOGO_PATH):
        logo = Image.open(LOGO_PATH).convert("RGBA")
        logo.thumbnail((int(zw * 0.55), int(S * 0.25)), Image.Resampling.LANCZOS)
        lay = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        lay.paste(logo, (cx - logo.width // 2, y), logo)
        shadow_glow(canvas, lay.getchannel("A"), glow=0.38, shadow=0.65)
        canvas.alpha_composite(lay)
        y += logo.height + int(S * 0.04)

    max_w = zw * 0.94
    sub_lines = copy.get("subtitle_deva_lines") if use_deva else []
    if not sub_lines:
        sub_lines = textwrap.wrap(copy.get("subtitle_en") or fest.get("subtitle_en", ""), 34)[:3]
    sub_role = "deva_body" if (use_deva and copy.get("subtitle_deva_lines")) else "latin"

    for sc in (1.0, 0.92, 0.85, 0.78, 0.70):
        t_role = "deva_title" if use_deva else "latin"
        title_text = copy["title_deva"] if use_deva else fest["festival_title_en"].upper()
        tf = fit_font(t_role, title_text, int(S * (0.15 if use_deva else 0.075) * sc), max_w)
        pf = font("deva_title" if use_deva else "latin", int(S * 0.052 * sc))
        sf = font(sub_role, int(S * 0.0345 * sc))
        for ln in sub_lines:
            while measure(ln, sf)[2] - measure(ln, sf)[0] > max_w and sf.size > 18:
                sf = font(sub_role, sf.size - 2)
        est = (y + (S * 0.06 * sc if use_deva and copy.get("title_prefix_deva") else 0) + tf.size * 1.25
               + S * 0.085 * sc + len(sub_lines) * sf.size * 1.65)
        if est <= zone[3] + S * 0.02:
            break

    # prefix (e.g. "शुभ"), a little left of centre like the reference posters
    tl, tt, tr, tb = measure(title_text, tf)
    title_w = tr - tl
    if use_deva and copy.get("title_prefix_deva"):
        m, bot = draw_text_mask(copy["title_prefix_deva"], pf, cx - title_w * 0.22, y)
        paint_metal(canvas, m, y, int(bot), stops, glow=0.25)
        y = int(bot + S * 0.006)

    m, bot = draw_text_mask(title_text, tf, cx, y)
    paint_metal(canvas, m, y, int(bot), stops, glow=0.42, shadow=0.7)
    y = int(bot + S * 0.032)

    orn = ornament_mask(cx, y + S * 0.018, min(zw * 0.34, S * 0.2))
    paint_metal(canvas, orn, y - int(S * 0.02), y + int(S * 0.05), stops, glow=0.25, shadow=0.5, bevel=False)
    y += int(S * 0.062)

    cream = (255, 244, 222)
    for ln in sub_lines:
        m, bot = draw_text_mask(ln, sf, cx, y)
        shadow_glow(canvas, m, glow=0.0, shadow=0.75)
        fill = Image.new("RGBA", (S, S), cream + (255,))
        fill.putalpha(m)
        canvas.alpha_composite(fill)
        y = int(bot + sf.size * 0.55)

    # footer + frame
    ff = font("latin", int(S * 0.0165))
    fm = tracked("SADHGURU TILES  \u2022  PREMIUM MARBLE, GRANITE & TILES", ff, S // 2, int(S * 0.952), int(S * 0.0035))
    paint_metal(canvas, fm, int(S * 0.94), int(S * 0.97), stops, glow=0.15, shadow=0.5, bevel=False)
    paint_metal(canvas, frame_mask(), 0, S, stops, glow=0.18, shadow=0.35, bevel=False)

    # film grain for a photographic feel, then lossless save
    arr = np.asarray(canvas.convert("RGB"), np.float32)
    arr += np.random.default_rng(3).normal(0, 1.8, (S, S, 1)).astype(np.float32)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(out_path, "PNG", compress_level=4)
    log(f"Saved {out_path}")


# ----------------------------------------------------------------------------
# Delivery
# ----------------------------------------------------------------------------
def wa_headers():
    return {"Authorization": f"Bearer {WA_ACCESS_TOKEN}"}


def wa_send_document(path, caption):
    data = Path(path).read_bytes()
    up = http("POST", f"https://graph.facebook.com/v22.0/{WA_PHONE_NUMBER_ID}/media", headers=wa_headers(),
              files={"file": (Path(path).name, data, "image/png"), "messaging_product": (None, "whatsapp"),
                     "type": (None, "image/png")}, timeout=120)
    media_id = up.json()["id"]
    payload = {"messaging_product": "whatsapp", "to": CLIENT_WA_NUMBER, "type": "document",
               "document": {"id": media_id, "filename": Path(path).name, "caption": caption}}
    http("POST", f"https://graph.facebook.com/v22.0/{WA_PHONE_NUMBER_ID}/messages", headers=wa_headers(), json=payload, timeout=60)


def wa_send_template():
    payload = {"messaging_product": "whatsapp", "to": CLIENT_WA_NUMBER, "type": "template",
               "template": {"name": WA_TEMPLATE_NAME, "language": {"code": WA_TEMPLATE_LANG}}}
    http("POST", f"https://graph.facebook.com/v22.0/{WA_PHONE_NUMBER_ID}/messages", headers=wa_headers(), json=payload, timeout=60)


def telegram_send(path, caption):
    http("POST", f"https://api.telegram.org/bot{TG_TOKEN}/sendDocument", data={"chat_id": TG_CHAT, "caption": caption[:1000]},
         files={"document": (Path(path).name, Path(path).read_bytes(), "image/png")}, timeout=120)


def build_caption(fest, idx, total):
    return (f"\u2728 *{fest['festival_title_en']}* ({fest['date']}) [{idx}/{total}]\n\n"
            f"\U0001FA94 *\u092E\u0930\u093E\u0920\u0940:* {fest['greeting_marathi']}\n\n"
            f"\U0001F1EE\U0001F1F3 *\u0939\u093F\u0928\u094D\u0926\u0940:* {fest['greeting_hindi']}")


def deliver(path, fest, idx, total):
    """Returns True if the poster itself reached at least one channel."""
    caption = build_caption(fest, idx, total)
    delivered = False
    if WA_ACCESS_TOKEN and WA_PHONE_NUMBER_ID and CLIENT_WA_NUMBER:
        try:
            wa_send_document(path, caption)
            log("  WhatsApp document sent")
            delivered = True
        except ApiError as e:
            log(f"  WhatsApp document failed: {e}")
            if "131047" in e.body or "131026" in e.body or "re-engage" in e.body.lower():
                try:
                    wa_send_template()
                    log("  24h window closed -> sent template nudge (ask client to reply 'Hi' to reopen)")
                except Exception as e2:
                    log(f"  template nudge failed: {e2}")
        except Exception as e:
            log(f"  WhatsApp failed: {e}")
    if TG_TOKEN and TG_CHAT:
        try:
            telegram_send(path, caption.replace("*", ""))
            log("  Telegram backup sent")
            delivered = True
        except Exception as e:
            log(f"  Telegram failed: {e}")
    return delivered


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    target, festivals = get_festivals()
    if not festivals:
        log(f"No festivals on {target}. Done.")
        return
    ensure_fonts()
    if not HAS_RAQM:
        log("WARNING: Pillow has no raqm; Devanagari disabled, English fallback typography will be used.")
    sent = load_sent()
    failures = 0
    OUT_DIR.mkdir(exist_ok=True)
    for idx, fest in enumerate(festivals, 1):
        key = f"{target}|{slugify(fest['festival_title_en'])}"
        if key in sent and not E("TEST_DATE") and not E("FORCE"):
            log(f"Already sent {key}; skipping (duplicate trigger).")
            continue
        try:
            side = fest.get("text_side", "right")
            tag = f"{target}_{idx}_{slugify(fest['festival_title_en'])}"
            copy = art_direct(fest, side)
            cands = make_candidates(copy["scene_prompt"], tag)
            best = pick_best(cands, fest) if len(cands) > 1 else cands[0][1]
            out = OUT_DIR / f"Sadhguru_Tiles_{tag}.png"
            compose(best, fest, copy, str(out), side)
            if DRY_RUN:
                log("DRY_RUN: not sending.")
                continue
            if deliver(str(out), fest, idx, len(festivals)):
                if not E("TEST_DATE"):
                    mark_sent(key)
            else:
                failures += 1
        except Exception:
            failures += 1
            log(f"FAILED {fest.get('festival_title_en')}:\n{traceback.format_exc()}")
    if failures:
        log(f"{failures} poster(s) failed -> exiting 1 so GitHub emails you.")
        sys.exit(1)


if __name__ == "__main__":
    main()
