from ultralytics import YOLO
from gtts import gTTS
from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import cohere
import cv2
import requests
 
response = requests.get('https://deepstackpython.readthedocs.io/en/latest/_images/test-image3.jpg')
image = Image.open(BytesIO(response.content))
image = np.asarray(image)

model = YOLO("yolov8n.pt")
co = cohere.Client('VgR2hXk1OC9UOiTWFYE1rTodw1GkT7xYKI6MsLIS') 
yoloLabels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

predict = model.predict(image, conf=0.25)
results = predict[0].boxes.data

talk1 = 'In this image, I can see '
talk2 = ''
detected = []

for result in results:
    x1 = int(result[0])
    y1 = int(result[1])
    x2 = int(result[2])
    y2 = int(result[3])
    confidence = result[4]
    label = yoloLabels[int(result[5])]

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = f"{label}: {confidence:.2f}"
    cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    detected.append(label)

cleared = []
count = []
for obj in detected:
    if obj not in cleared:
        cleared.append(obj)

for object in cleared:
    count = str(detected.count(object))
    talk1 = talk1 + count + ' ' + object + ', '
    response = co.generate(
        model='command-nightly',
        prompt='how to draw ' + object,
        max_tokens=300,
        temperature=0.9,
        k=0,
        stop_sequences=[],
        return_likelihoods='NONE')
    talk2 = talk2 + f'To draw {object}, {response.generations[0].text} '

mytext = talk1 + talk2
print(mytext)
audio = gTTS(text=mytext, lang="en", slow=False)
audio.save("drawingguide.mp3")

plt.imshow(image)
plt.axis('off')
plt.show()