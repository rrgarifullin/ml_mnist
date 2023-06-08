import torch
from model import Net
from fastapi import FastAPI, File, UploadFile, Request, Form
from PIL import Image
import io
from predict import predict
import numpy
from fastapi.templating import Jinja2Templates


model = Net()

# train_model(model)

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/upload")
async def upload_image(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    with open(file.filename, "wb") as f:
        f.write(contents)
    img = Image.open(io.BytesIO(contents))
    img = img.convert('L').resize((28, 28))
    img = 1 - (numpy.array(img) / 255.0)
    img = torch.from_numpy(img).float()
    digit = predict(img)
    return templates.TemplateResponse("result.html", {"request": request, "filename": file.filename, "digit": digit})
