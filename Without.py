# execute "pip install ultralytics" on terminal
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import cv2
import requests

model = YOLO("yolov8n.pt")

response = requests.get("https://deepstackpython.readthedocs.io/en/latest/_images/test-image3.jpg")
image = Image.open(BytesIO(response.content))

image = np.asarray(image)

results = model.predict(image, conf=0.25)
list = results[0].boxes.data

for result in list:
    print(result[0])


#plt.imshow(image)
#ax = plt.gca()
#ax.get_xaxis().set_visible(False)
#ax.get_yaxis().set_visible(False)
#plt.show()


