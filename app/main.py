from fastapi import FastAPI
from dotenv import load_dotenv
import os
from fastapi import UploadFile, File, Form
from fastapi.responses import JSONResponse
from openai import OpenAI
from fastapi.responses import HTMLResponse
import imageio_ffmpeg
import tempfile, subprocess, uuid, shutil


# Загружаем переменные из .env
load_dotenv()

app = FastAPI()


# клиент OpenAI (использует ключ из .env)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# карты языков
LANG_NAME = {"ru":"Russian", "en":"English", "de":"German"}
LANG_CODE = {"ru":"ru", "en":"en", "de":"de"}  # коды для STT
def name_of(code): return {"ru":"Русский","en":"English","de":"Deutsch"}.get(code, "Auto")



@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(os.path.dirname(__file__), "templates", "index.html"), encoding="utf-8") as f:
        return f.read()
    


@app.post("/speech-translate")
async def speech_translate(
    file: UploadFile = File(...),
    source_lang: str = Form("auto"),   # 'auto' | 'ru' | 'en' | 'de'
    target_lang: str = Form("en"),     # 'ru' | 'en' | 'de'
):
    # 1) сохраняем входной файл
    suffix = os.path.splitext(file.filename)[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        data = await file.read()
        tmp_in.write(data)
        in_path = tmp_in.name

    # 2) конвертируем в WAV 16kHz mono
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
        # 3) STT (можно попробовать whisper-1 если точность нужна максимальная)
        stt_kwargs = {"model": "gpt-4o-transcribe"}
        if source_lang in LANG_CODE:  # задаём язык явно
            stt_kwargs["language"] = LANG_CODE[source_lang]

        with open(out_path, "rb") as f:
            stt = client.audio.transcriptions.create(file=f, **stt_kwargs)

        src_text = (stt.text or "").strip()
        detected = getattr(stt, "language", None)  # может быть None
        src_code = source_lang if source_lang in LANG_CODE else (detected or "auto")

        # 4) Перевод (если target == source — просто возвращаем исходник)
        if target_lang == src_code:
            translated = src_text
        else:
            tgt_name_en = LANG_NAME[target_lang]  # имя на английском для подсказки модели
            prompt = f"Translate the following text to {tgt_name_en}. Keep meaning, names and numbers:\n\n{src_text}"
            comp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":"You are a precise, context-aware translation engine."},
                    {"role":"user","content":prompt},
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




@app.post("/transcribe")
async def transcribe(file: UploadFile):
    try:
        # 1. Сохраняем входной webm во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_in:
            data = await file.read()
            tmp_in.write(data)
            tmp_in_path = tmp_in.name

        # 2. Создаём временный wav-файл
        tmp_out_path = tmp_in_path.replace(".webm", ".wav")

        # 3. Конвертация webm → wav через ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-i", tmp_in_path,
            "-ar", "16000", "-ac", "1", "-f", "wav",
            tmp_out_path
        ]
        subprocess.run(cmd, check=True)

        # 4. Отправляем wav в OpenAI Whisper
        with open(tmp_out_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",   # или whisper-1
                file=audio_file,
                language="ru"                # ← явно указываем русский
            )

        # 5. Чистим временные файлы
        os.remove(tmp_in_path)
        os.remove(tmp_out_path)

        return {"text": transcript.text, "source_language": "auto"}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

