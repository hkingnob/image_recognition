# Web app that recognizes basic images - just so I get the hang of it!

import os
import base64
import json
import re
from flask import Flask, render_template, request
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o-mini")
MAX_IMAGE_MB = int(os.getenv("MAX_IMAGE_MB", "8"))

# OpenAI client
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Creates the Flask app
def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        # We render the same page for form + (optional) results
        return render_template("index.html", results=None, error=None)
    
    @app.post("/analyze")
    def analyze():
        if "image" not in request.files:
            return render_template("index.html", results=None, error="No file uploaded"), 400
        
        file = request.files["image"]
        if file.filename == "":
            return render_template("index.html", results=None, error="Empty filename"), 400
        
        try:
            image_bytes = _read_image(file, max_mb=MAX_IMAGE_MB)
        except ValueError as e:
            return render_template("index.html", results=None, error=str(e)), 400
        
        try:
            results = call_gpt(image_bytes, model=VISION_MODEL)
        except Exception as e:
            return render_template("index.html", results=None, error=f"Model error: {e}"), 502
        
        return render_template("index.html", results=results, error=None)

    @app.get("/healthz")
    def healthz():
        return {"ok": True}
    
    return app


def _read_image(file_storage, max_mb: int) -> bytes:
    # Minimal size check for now
    raw = file_storage.read()
    if not raw:
        raise ValueError("Uploaded file was empty.")
    
    size_mb = len(raw) / (1024 * 1024)
    if size_mb > max_mb:
        raise ValueError(f"Image too large ({size_mb:.1f} MB). Limit is {max_mb} MB.")
    
    return raw

def _parse_json_safely(text: str) -> dict | None:
    """Try to parse JSON; tolerate ```json ... ``` or ``` ... ``` fences."""
    t = text.strip()
    # remove leading ```json or ``` and trailing ```
    t = re.sub(r"^\s*```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"\s*```\s*$", "", t)
    try:
        return json.loads(t)
    except Exception:
        return None

def call_gpt(image_bytes: bytes, model: str) -> dict:
    # Minimal helper that sends one image to a vision-capable GPT model and expects strict JSON back

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set.")
    
    # Prepare a data: URL for widest compatibility
    
    mime = _sniff_mime(image_bytes)
    data_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('utf-8')}"

    system_prompt = (
        "You are a vision model that identifies objects in a single image. "
        "Return JSON ONLY (no markdown, no backticks, no code fences). The JSON must be of the form: "
        "{\"objects\":[{\"name\": string, \"confidence\": number (0..1)}], "
        "\"summary\": string}. "
        "Be concise, avoid speculation; include lower-confidence items with a lower confidence."
    )

    user_prompt = "Identify the salient objects in this image and summarize in one sentence."

    # Compose a chat with an image input (data URL)
    # Newer clients accept content parts with image_url dicts.
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url}
                    },
                ],
            },
        ]
    )

    content = response.choices[0].message.content.strip()

    # Parse JSON, tolerating fenced markdown if the model ignored instructions
    data = _parse_json_safely(content)
    if data is None:
        data = {"objects": [], "summary": content}

    # Normalize minimal shape (ie if it's returned in a dumb form from GPT)
    if "objects" not in data or not isinstance(data["objects"], list):
        data["objects"] = []
    if "summary" not in data or not isinstance(data["summary"], str):
        data["summary"] = ""

    # Clamp confidences if present
    for obj in data["objects"]:
        if isinstance(obj, dict) and "confidence" in obj:
            try:
                c = float(obj["confidence"])
                obj["confidence"] = max(0.0, min(1.0, c))
            except Exception:
                obj["confidence"] = None
    
    return data


# Returns the appropriate file type
def _sniff_mime(b: bytes) -> str:
    # JPEG
    if len(b) > 2 and b[0:2] == b"\xff\xd8":
        return "image/jpeg"
    # PNG
    if len(b) > 8 and b[0:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    # WebP
    if len(b) > 12 and b[0:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "image/webp"
    # GIF
    if len(b) > 6 and (b[0:6] in (b"GIF87a", b"GIF89a")):
        return "image/gif"
    # Fallback
    return "application/octet-stream"

if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=True)