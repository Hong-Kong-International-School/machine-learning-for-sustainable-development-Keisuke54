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

predict = model.predict(image, conf=0.25)
results = predict[0].boxes.data

for result in results:
    x1 = int(result[0])
    y1 = int(result[1])
    x2 = int(result[2])
    y2 = int(result[3])
    label = result[4]
    confidence = result[5]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = f"{label}: {confidence:.2f}"
    cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


plt.imshow(image)
plt.axis('off')
plt.show()