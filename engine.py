import os
import json
import base64
import textwrap
from io import BytesIO
from datetime import datetime, timedelta, timezone
import requests
from PIL import Image, ImageDraw, ImageFont

# 1. Load Environment Secrets
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID")
CF_API_TOKEN = os.getenv("CF_API_TOKEN")
WA_PHONE_NUMBER_ID = os.getenv("WA_PHONE_NUMBER_ID")
WA_ACCESS_TOKEN = os.getenv("WA_ACCESS_TOKEN")
CLIENT_WA_NUMBER = os.getenv("CLIENT_WA_NUMBER")

FESTIVAL_FILE = "data/festivals_2026_2027.json"
LOGO_PATH = "assets/sadhguru_tiles_logo.png"


def get_tomorrows_festivals():
    """Finds all festivals scheduled for TOMORROW (IST) so posters arrive at 7:30 PM the day before."""
    ist = timezone(timedelta(hours=5, minutes=30))
    tomorrow_ist = datetime.now(ist) + timedelta(days=1)
    target_date = tomorrow_ist.strftime("%Y-%m-%d")
    target_year = tomorrow_ist.strftime("%Y")

    # Allow manual override via GitHub Actions test_date input
    override_date = os.getenv("TEST_DATE", "").strip()
    if override_date:
        target_date = override_date
        target_year = override_date.split("-")[0]

    print(f"Checking festivals for target date: {target_date}")

    with open(FESTIVAL_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    year_list = data.get(target_year, [])
    matches = [item for item in year_list if item.get("date") == target_date]
    return target_date, matches


def enhance_prompt_with_gemini(festival):
    """Uses Gemini 2.5 Flash to craft a rich, creative commercial prompt for FLUX."""
    base_prompt = festival["visual_prompt"]
    title_en = festival.get("festival_title_en", "INDIAN FESTIVAL")

    if not GEMINI_API_KEY:
        return f"{base_prompt}, no text, no watermark, ultra-detailed 8k commercial photography, clean dark lower third for branding"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    system_instruction = (
        "You are an award-winning luxury brand art director for 'Sadhguru Tiles' (premium marble, granite & ceramic tiles). "
        "Expand the given festival visual concept into a single, vivid 80-word image generation prompt for FLUX.1-schnell. "
        "Include rich traditional Indian festive elements, cinematic volumetric lighting, and subtle luxury marble/ceramic tile reflections. "
        "CRITICAL: Leave the bottom 25% of the frame slightly dark/clean for text overlay. Strictly specify NO text, NO letters, NO watermarks inside the generated image."
    )
    payload = {
        "contents": [{
            "parts": [{"text": f"Festival: {title_en}\nConcept: {base_prompt}\nWrite only the raw image prompt:"}]
        }],
        "systemInstruction": {"parts": [{"text": system_instruction}]}
    }

    try:
        res = requests.post(url, json=payload, timeout=20)
        res.raise_for_status()
        return res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"Gemini fallback used ({e})")
        return f"{base_prompt}, luxury marble tile reflections, cinematic golden lighting, dark clean lower third, no text, 8k photorealistic"


def generate_flux_image(prompt):
    """Calls Cloudflare Workers AI FLUX-1-schnell and returns a 1080x1080 PIL Image."""
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/@cf/black-forest-labs/flux-1-schnell"
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}"}
    payload = {"prompt": prompt, "steps": 6}

    res = requests.post(url, headers=headers, json=payload, timeout=60)
    res.raise_for_status()
    result = res.json()

    b64_img = result["result"]["image"]
    img_bytes = base64.b64decode(b64_img)
    img = Image.open(BytesIO(img_bytes)).convert("RGBA")
    return img.resize((1080, 1080), Image.Resampling.LANCZOS)


def load_english_font(size, bold=True):
    """Loads a clean English font from Ubuntu system fonts."""
    candidates = [
        "assets/Montserrat-Bold.ttf" if bold else "assets/Montserrat-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def apply_creative_english_branding(base_img, festival, output_path):
    """Applies a luxury bottom gradient, double gold frame border, English typography, and Sadhguru Tiles logo."""
    W, H = base_img.size

    # 1. Smooth dark luxury gradient over bottom 42% for crisp text legibility
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)

    for y in range(int(H * 0.58), H):
        progress = (y - int(H * 0.58)) / (H - int(H * 0.58))
        alpha = int(235 * (progress ** 1.35))
        draw_ov.line([(0, y), (W, y)], fill=(10, 8, 6, alpha))

    composited = Image.alpha_composite(base_img, overlay)
    draw = ImageDraw.Draw(composited)

    # 2. Double luxury gold outer border
    gold = (218, 178, 95, 220)
    light_gold = (248, 225, 160, 255)
    white = (255, 255, 255, 255)
    draw.rectangle([28, 28, W - 28, H - 28], outline=gold, width=3)
    draw.rectangle([38, 38, W - 38, H - 38], outline=(218, 178, 95, 90), width=1)

    # 3. Stamp Sadhguru Tiles Logo at Top-Right
    if os.path.exists(LOGO_PATH):
        logo = Image.open(LOGO_PATH).convert("RGBA")
        logo.thumbnail((230, 120), Image.Resampling.LANCZOS)
        lw, lh = logo.size
        composited.paste(logo, (W - lw - 60, 58), logo)

    # 4. English Festival Title + Creative Subtitle + Brand Footer
    font_title = load_english_font(52, bold=True)
    font_sub = load_english_font(27, bold=False)
    font_brand = load_english_font(25, bold=True)

    title_en = festival.get("festival_title_en", "FESTIVAL GREETINGS").upper()
    subtitle_en = festival.get("subtitle_en", "Wishing you peace, prosperity & timeless elegance.")

    # Decorative top accent line
    draw.line([(W // 2 - 90, 775), (W // 2 + 90, 775)], fill=gold, width=3)

    # Auto-scale Title if long
    tb = draw.textbbox((0, 0), title_en, font=font_title)
    tw = tb[2] - tb[0]
    if tw > W - 120:
        font_title = load_english_font(40, bold=True)
        tb = draw.textbbox((0, 0), title_en, font=font_title)
        tw = tb[2] - tb[0]
    draw.text(((W - tw) // 2, 795), title_en, font=font_title, fill=light_gold)

    # Wrapped Subtitle
    wrapped_sub = textwrap.wrap(subtitle_en, width=52)
    y_cursor = 870
    for line in wrapped_sub[:2]:
        sb = draw.textbbox((0, 0), line, font=font_sub)
        sw = sb[2] - sb[0]
        draw.text(((W - sw) // 2, y_cursor), line, font=font_sub, fill=white)
        y_cursor += 36

    # Bottom Brand Signature Bar
    brand_line = "SADHGURU TILES  •  PREMIUM MARBLE, GRANITE & TILES"
    bb = draw.textbbox((0, 0), brand_line, font=font_brand)
    bw = bb[2] - bb[0]
    draw.line([(120, 968), (W - 120, 968)], fill=(218, 178, 95, 110), width=1)
    draw.text(((W - bw) // 2, 986), brand_line, font=font_brand, fill=gold)

    composited.convert("RGB").save(output_path, "PNG", optimize=False, compress_level=0)
    print(f"Saved branded poster: {output_path}")


def send_whatsapp_document(file_path, festival, index, total):
    """Uploads the PNG to Meta Cloud API and sends it as an uncompressed Document."""
    media_url = f"https://graph.facebook.com/v22.0/{WA_PHONE_NUMBER_ID}/media"
    headers = {"Authorization": f"Bearer {WA_ACCESS_TOKEN}"}

    with open(file_path, "rb") as f:
        files = {
            "file": (os.path.basename(file_path), f, "image/png"),
            "messaging_product": (None, "whatsapp"),
        }
        upload_res = requests.post(media_url, headers=headers, files=files, timeout=60)
        upload_res.raise_for_status()
        media_id = upload_res.json()["id"]

    msg_url = f"https://graph.facebook.com/v22.0/{WA_PHONE_NUMBER_ID}/messages"
    caption = (
        f"✨ *{festival['festival_title_en']}* ({festival['date']}) [{index}/{total}]\n\n"
        f"🇲🇷 *Marathi:* {festival['greeting_marathi']}\n\n"
        f"🇮🇳 *Hindi:* {festival['greeting_hindi']}"
    )
    payload = {
        "messaging_product": "whatsapp",
        "to": CLIENT_WA_NUMBER,
        "type": "document",
        "document": {
            "id": media_id,
            "filename": os.path.basename(file_path),
            "caption": caption,
        },
    }
    msg_res = requests.post(msg_url, headers=headers, json=payload, timeout=30)
    msg_res.raise_for_status()
    print(f"Sent WhatsApp document [{index}/{total}]: {os.path.basename(file_path)}")


def main():
    target_date, festivals = get_tomorrows_festivals()
    if not festivals:
        print(f"No festivals scheduled for target date ({target_date}). Exiting cleanly.")
        return

    print(f"Found {len(festivals)} festival(s) for {target_date}. Generating posters...")
    for idx, fest in enumerate(festivals, start=1):
        slug = fest["festival_title_en"].lower().replace(" ", "_").replace("'", "")
        out_name = f"Sadhguru_Tiles_{target_date}_{idx}_{slug}.png"

        prompt = enhance_prompt_with_gemini(fest)
        raw_img = generate_flux_image(prompt)
        apply_creative_english_branding(raw_img, fest, out_name)
        send_whatsapp_document(out_name, fest, idx, len(festivals))


if __name__ == "__main__":
    main()
