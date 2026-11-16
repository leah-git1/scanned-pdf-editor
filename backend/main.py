from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF
import easyocr
import numpy as np
import cv2
import base64
import io

app = FastAPI(title="Scanned PDF Editor API")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize EasyOCR reader with English support
reader = easyocr.Reader(['en'], gpu=False)

class EditRequest(BaseModel):
    image_base64: str
    box: dict  # {"x": int, "y": int, "width": int, "height": int}
    new_text: str

@app.post("/api/analyze-pdf")
async def analyze_pdf(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    # Process the first page as an example
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    
    # Convert to RGB format for OpenCV and EasyOCR
    if pix.n == 4:
        img_rgb = cv2.cvtColor(img_data, cv2.COLOR_RGBA2RGB)
    else:
        img_rgb = img_data

    # Run OCR text detection
    results = reader.readtext(img_rgb)
    
    detected_items = []
    for (bbox, text, prob) in results:
        top_left = [int(bbox[0][0]), int(bbox[0][1])]
        bottom_right = [int(bbox[2][0]), int(bbox[2][1])]
        
        detected_items.append({
            "text": text,
            "confidence": round(float(prob), 2),
            "box": {
                "x": top_left[0],
                "y": top_left[1],
                "width": bottom_right[0] - top_left[0],
                "height": bottom_right[1] - top_left[1]
            }
        })

    # Encode image to Base64 for frontend preview
    _, buffer = cv2.imencode('.png', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "image": f"data:image/png;base64,{img_base64}",
        "width": pix.w,
        "height": pix.h,
        "words": detected_items
    }

@app.post("/api/edit-text")
async def edit_text(req: EditRequest):
    # Decode the base64 image
    img_data = base64.b64decode(req.image_base64.split(",")[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    x = req.box["x"]
    y = req.box["y"]
    w = req.box["width"]
    h = req.box["height"]

    # 1. Inpaint to remove the old text cleanly
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    padding = 2
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(img.shape[1], x + w + padding)
    y_end = min(img.shape[0], y + h + padding)
    
    mask[y_start:y_end, x_start:x_end] = 255
    restored_img = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # 2. Render new text using Pillow
    pil_img = Image.fromarray(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font = ImageFont.truetype("arial.ttf", size=int(h * 0.8))
    except IOError:
        font = ImageFont.load_default()

    draw.text((x, y), req.new_text, fill=(0, 0, 0), font=font)

    # Convert back to base64
    result_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', result_bgr)
    updated_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "updated_image": f"data:image/png;base64,{updated_base64}"
    }