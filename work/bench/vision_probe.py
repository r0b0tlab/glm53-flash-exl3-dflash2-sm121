#!/usr/bin/env python3
"""Vision smoke for the published config: two synthetic images (text-in-image
and multi-object count/color) through /v1/chat/completions. Prints the model's
readings; used for the docs/RESULTS.md vision row."""
import base64
import io
import json
import urllib.request

from PIL import Image, ImageDraw

MODEL = "glm53-flash-exl3-dflash2"
BASE = "http://127.0.0.1:8000"


def ask(prompt, img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}}]}],
            "temperature": 0, "max_tokens": 256,
            "chat_template_kwargs": {"reasoning_effort": "low"}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    return d["choices"][0].get("finish_reason"), d["choices"][0]["message"].get("content")


def img_text():
    img = Image.new("RGB", (384, 256), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([40, 60, 344, 200], outline="red", width=6)
    d.text((120, 110), "BANANA 42", fill="black")
    return img


def img_shapes():
    img = Image.new("RGB", (400, 300), "white")
    d = ImageDraw.Draw(img)
    for i, col in enumerate(("blue", "green", "orange")):
        d.ellipse([30 + i * 120, 100, 110 + i * 120, 180], fill=col)
    d.text((30, 240), "HELLO WORLD", fill="black")
    return img


def main():
    fin, text = ask("What word and number are in the image?", img_text())
    print("probe1 finish=%s: %r" % (fin, (text or "")[:200]), flush=True)
    fin, text = ask("How many circles are in the image, what are their colors, "
                    "and what text is written at the bottom?", img_shapes())
    print("probe2 finish=%s: %r" % (fin, (text or "")[:300]), flush=True)
    print("VISION_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
