# execute "pip install ultralytics" on terminal
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import cv2
import requests

model = YOLO("yolov8n.pt")

response = requests.get("https://static.photocdn.pt/images/articles/2019/10/02/Simple_Landscape_Photography_Tips_With_Tons_of_Impact.jpg")
image = Image.open(BytesIO(response.content))

image = np.asarray(image)

results = model.predict(image, conf=0.25)

print(results[0].boxes.boxes)

plt.imshow(image)
#get current axes
ax = plt.gca()
#hide x-axis
ax.get_xaxis().set_visible(False)
#hide y-axis 
ax.get_yaxis().set_visible(False)
plt.show()