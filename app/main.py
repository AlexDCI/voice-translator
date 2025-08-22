from fastapi import FastAPI, UploadFile, File, Form, Depends, Header, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
import os, tempfile, subprocess
import imageio_ffmpeg
import jwt

# Загружаем переменные из .env
load_dotenv()

app = FastAPI()


# клиент OpenAI (использует ключ из .env)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


VOICE_JWT_SECRET = os.getenv("VOICE_JWT_SECRET", "change-me")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("DJANGO_ORIGIN", "http://127.0.0.1:8000")],  # твой Django
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_user(authorization: str | None = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, VOICE_JWT_SECRET, algorithms=["HS256"])
        return {"user_id": payload["user_id"], "username": payload["username"]}
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


# карты языков
# карты языков: code -> English name
LANG_NAME = {
    "en": "English",
    "ru": "Russian",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "zh": "Chinese",
    "zh-tw": "Chinese-Traditional",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "bn": "Bengali",
    "ur": "Urdu",
    "tr": "Turkish",
    "nl": "Dutch",
    "el": "Greek",
    "pl": "Polish",
    "cs": "Czech",
    "hu": "Hungarian",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "he": "Hebrew",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "fil": "Filipino",
    "ro": "Romanian",
    "sk": "Slovak",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "sr": "Serbian",
    "sl": "Slovenian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "et": "Estonian",
    "ka": "Georgian",
    "hy": "Armenian",
    "fa": "Persian",
    "ps": "Pashto",
    "az": "Azerbaijani",
    "kk": "Kazakh",
    "uz": "Uzbek",
    "tg": "Tajik",
    "tk": "Turkmen",
    "ky": "Kyrgyz",
    "mn": "Mongolian",
    "sw": "Swahili",
    "zu": "Zulu",
    "xh": "Xhosa",
    "af": "Afrikaans",
    "ht": "Haitian Creole",
    "eu": "Basque",
    "gl": "Galician",
    "ca": "Catalan",
    "ga": "Irish",
    "cy": "Welsh",
    "gd": "Scottish Gaelic",
    "mt": "Maltese",
    "is": "Icelandic",
    "sa": "Sanskrit",
    "bo": "Tibetan",
    "mi": "Maori",
    "sm": "Samoan",
    "to": "Tongan"
}

def name_of(code: str) -> str:
    """Красивое название языка для ответа"""
    if code == "auto":
        return "Auto"
    return LANG_NAME.get(code, code)


@app.post("/speech-translate")
async def speech_translate(
    file: UploadFile = File(...),
    source_lang: str = Form("auto"),   # 'auto' | 'en' | 'ru' | ...
    target_lang: str = Form("en"),     # 'en' | 'ru' | ...
    user=Depends(get_user)
):
    # сохраняем входной файл
    suffix = os.path.splitext(file.filename)[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        data = await file.read()
        tmp_in.write(data)
        in_path = tmp_in.name

    out_path = in_path.replace(suffix, ".wav")
    try:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [ffmpeg, "-y", "-i", in_path, "-ar", "16000", "-ac", "1", "-f", "wav", out_path]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        try: os.remove(in_path)
        except: pass
        return JSONResponse({"error": f"FFmpeg convert error: {e}"}, status_code=500)

    try:
        # === 1) Speech-to-text (STT) ===
        stt_kwargs = {"model": "gpt-4o-transcribe"}
        if source_lang in LANG_NAME:  # язык явно указан
            stt_kwargs["language"] = source_lang  # ← передаём код (например "ru"), а не "Russian"

        with open(out_path, "rb") as f:
            stt = client.audio.transcriptions.create(file=f, **stt_kwargs)

        src_text = (stt.text or "").strip()
        detected = getattr(stt, "language", None)  # язык, который определила модель
        src_code = source_lang if source_lang in LANG_NAME else (detected or "auto")

        # === 2) Перевод ===
        if target_lang == src_code:
            translated = src_text
        else:
            if target_lang not in LANG_NAME:
                target_lang = "en"  # fallback
            tgt_name_en = LANG_NAME[target_lang]
            prompt = f"Translate the following text to {tgt_name_en}. Keep meaning, names and numbers:\n\n{src_text}"
            comp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a precise, context-aware translation engine."},
                    {"role": "user", "content": prompt},
                ],
            )
            translated = comp.choices[0].message.content.strip()

        return {
            "source_language": src_code,
            "source_language_name": name_of(src_code),
            "target_language": target_lang,
            "target_language_name": name_of(target_lang),
            "source_text": src_text,
            "translated_text": translated,
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        for p in (in_path, out_path):
            try: os.remove(p)
            except: pass
