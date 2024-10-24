import cv2
import matplotlib.pyplot as plt
from tkinter import filedialog


imgPath = filedialog.askopenfilename()
img = cv2.imread(imgPath)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face = face_classifier.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 7, minSize = (40, 40))

for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 6)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#cv2.imshow('face1', img)
#cv2.waitKey(0)

plt.figure(figsize = (20, 10))
plt.imshow(img_rgb)
plt.axis('off')



