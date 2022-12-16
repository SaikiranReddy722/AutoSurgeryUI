from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import base64
from base64 import b64encode
from json import dumps
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
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

model = tf.keras.models.load_model("C:/Users/SAI KIRAN REDDY/OneDrive/Desktop/resnet50_backbone.hdf5")

# CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

res="C:/Users/SAI KIRAN REDDY/OneDrive/Desktop/autosurgery-main/potato-disease-classification-main/uploads/result.png"
mypic="C:/Users/SAI KIRAN REDDY/OneDrive/Desktop/autosurgery-main/potato-disease-classification-main/uploads/myphoto.png"
he1="C:/Users/SAI KIRAN REDDY/OneDrive/Desktop/autosurgery-main/potato-disease-classification-main/frontend/src/help.png"
he2="C:/Users/SAI KIRAN REDDY/OneDrive/Desktop/autosurgery-main/potato-disease-classification-main/uploads/help.png"
img_class={
    0: (127,127,127),
    1: (200 ,143, 142),
    2: (238 ,122 ,118),
    3: (213 ,83 ,153),
    4: (186 ,183 ,92),
    5: (189, 253, 80),
    6: (236 ,98 ,43),
    7: (234, 51, 35),
    8: (255 ,255 ,84),
    9: (188, 253, 190),
    10: (242 ,164 ,167),
    11: (17 ,49, 123),
    12: (106 ,76 ,21),
    13: (255, 255, 255)
}



@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def new_annotations(numpydata):
    print(numpydata)
    new_np = np.zeros((len(numpydata),len(numpydata[0]),3)).astype('uint8')
    print(len(numpydata))
    for i in range(len(numpydata)):
        for j in range(len(numpydata[0])):
            val=numpydata[i][j]
            # print(val)
            new_np[i][j]=img_class[val]
    img =cv2.cvtColor(new_np, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (854, 480))
    # return img
    cv2.imwrite(res,img)
    return img

def result(x):
    m=cv2.imread(mypic)
# m=cv2.resize(m, (128, 128))
    o=cv2.imread(res)
    inp = np.zeros((480,854,3)).astype('uint8')
    for x in range(480):
        for y in range(854):
            inp[x][y][0]=m[x][y][0]/2+o[x][y][0]/2
            inp[x][y][1]=m[x][y][1]/2+o[x][y][1]/2
            inp[x][y][2]=m[x][y][2]/2+o[x][y][2]/2
    # img =cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    img = cv2.resize(inp, (854, 480))
    cv2.imwrite(he1,img)
    print("hello2")
    return img

def read_file_as_image(data):
    image = Image.open(BytesIO(data))
    print(image.size)
    image.save(mypic)
    # image = n p.array(image)
    # print(image.shape)
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    read_file_as_image(await file.read())

    test_img = cv2.imread(mypic)
    test_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2RGB)
    test_img = cv2.resize(test_img, (128, 128))
    test_img = Image.fromarray(test_img)
    test_img = np.array(test_img)
    test_img_input=np.expand_dims(test_img, 0)
    prediction = (model.predict(test_img_input))
    predicted_img=np.argmax(prediction, axis=3)[0,:,:]
    print(predicted_img)

    print("hello1")

    img=new_annotations(predicted_img)
    img=result('x')
    # img=cv2.imread('/Users/rishisaginala/Downloads/potato-disease-classification-main/uploads/help.png')
    with open(he2, 'rb') as open_file:
        byte_content = open_file.read()
    base64_bytes = b64encode(byte_content)
    base64_string = base64_bytes.decode('utf-8')
    raw_data = {"image": base64_string} 
    json_data = dumps(raw_data, indent=2)
    return {
        'class': json_data,
        'confidence': float(0)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=3001)

