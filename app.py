from fastapi import FastAPI,Request,UploadFile,File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app=FastAPI()
templates=Jinja2Templates(directory='templates')
model=tf.keras.models.load_model('ai_vs_real_human_classification_model.keras')
class_names=['AI','Real']
def load_image(image:Image.Image):
   image=image.convert('RGB')
   image=image.resize((224,224))
   image=np.array(image)/255.0
   image=np.expand_dims(image,axis=0)
   return image

@app.get("/",response_class=HTMLResponse)
def home(request:Request):
   return templates.TemplateResponse("index.html",{"request":request})

@app.post("/predict",response_class=HTMLResponse)
async def predict(request:Request,file:UploadFile=File(...)):
   con=await file.read()
   image=Image.open(io.BytesIO(con))
   img=load_image(image)
   prob=model.predict(img)[0][0]
   if prob >= 0.5:
       label="Real"
       confidence=prob
   else:
       label="AI"
       confidence=1-prob
   return templates.TemplateResponse(
        "index.html",
        {
            "request":request,
            "prediction":label,
            "confidence":f"{confidence*100:.2f}%"
        }
    )