# Web app/api related imports
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# model/image preprocessing related imports
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf


app = FastAPI()

app.mount("/static", StaticFiles(directory="../frontend/static"), name="static")
templates = Jinja2Templates(directory="../frontend/templates")

# Potato Crop Disease Classification model
PCDC_MODEL = tf.keras.models.load_model("../models/pcdc/PCDC_1")
PCDC_CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Pepper Bell Crop Disease Classification model
PBCDC_MODEL = tf.keras.models.load_model("../models/pbcdc/PBCDC_1")
PBCDC_CLASS_NAMES = ["Bacterial Spot", "Healthy"]

# function for processing the uploaded image
def img_bytes_to_img_array(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = image.resize((256,256),Image.ANTIALIAS)
    return np.array(image)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/potato", response_class=HTMLResponse)
async def potato(request: Request):
    return templates.TemplateResponse("pcdc.html", {"request": request})

# Potato Crop Disease Classification Model Api
@app.post("/potato/predict_pcdc", response_class=HTMLResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...)
):
    img = img_bytes_to_img_array(await file.read())
    img_batch = np.expand_dims(img, 0)

    prediction = PCDC_MODEL.predict(img_batch)
    predicted_status = PCDC_CLASS_NAMES[np.argmax(prediction[0])]
    confidence = float(np.max(prediction[0]))

    return templates.TemplateResponse(
        "pcdc.html",
        {"request": request,
        "status": predicted_status,
        "confidence": round(confidence, 2)}
        )
    
@app.get("/pepper_bell", response_class=HTMLResponse)
async def pepper_bell(request: Request):
    return templates.TemplateResponse("pbcdc.html", {"request": request})

# Pepper Bell Crop Disease Classification model Api
@app.post("/pepper_bell/predict_pbcdc", response_class=HTMLResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...)
):
    img = img_bytes_to_img_array(await file.read())
    img_batch = np.expand_dims(img, 0)

    prediction = PBCDC_MODEL.predict(img_batch)
    predicted_status = PBCDC_CLASS_NAMES[np.argmax(prediction[0])]
    confidence = float(np.max(prediction[0]))

    return templates.TemplateResponse(
        "pbcdc.html",
        {"request": request,
        "status": predicted_status,
        "confidence": round(confidence, 2)}
        )


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)