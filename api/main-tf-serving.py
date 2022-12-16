
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("/Users/rishisaginala/Downloads/potato-disease-classification-main/potatoes.h5")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data):
    image = Image.open(BytesIO(data))
    print(image.size)
    image.save('/Users/rishisaginala/Downloads/potato-disease-classification-main/uploads/myphoto.jpg')
    # image = n p.array(image)
    # print(image.shape)
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image=cv2.imread("/Users/rishisaginala/Downloads/potato-disease-classification-main/uploads/myphoto.jpg")
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    image=cv2.resize(image, (256, 256))
    image = Image.fromarray(image)
    image = np.array(image)
    # image=np.array(image)
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(100)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=3001)
