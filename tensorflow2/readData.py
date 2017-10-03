import glob
import time

import cv2
import csv

from tensorflow2.cnn import CnnObj
from tensorflow2.constants import EMOTIONS

import numpy as np

cap = cv2.VideoCapture('./data/075/075_11.mp4')

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

startTime = time.time()

counter = 0
csvFile = open('075_1.csv', 'w', newline='')
csvFileWriter = csv.writer(csvFile)
EMOTIONS.insert(0, 'second')
csvFileWriter.writerows([EMOTIONS])

fullStartTime = startTime
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1,
                                         minNeighbors=10, minSize=(5, 5),
                                         flags=cv2.CASCADE_SCALE_IMAGE)

    font = cv2.FONT_HERSHEY_SIMPLEX
    endTime = time.time()

    if (time.time() - startTime) > 0.5:

        counter += 1

        for (x, y, w, h) in faces:
            # remove wrong face
            if w * h < 100 * 100:
                continue

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            grayCut = gray[y:y + h, x:x + w]
            image = cv2.resize(grayCut, (128, 128))

            cnnObj = CnnObj()
            predList = cnnObj.getPredList(image)
            predList = np.insert(predList, 0, 0.5*counter)

            csvFileWriter.writerows([predList])

            # cv2.putText(frame, text, (x, y + h + 20), font, 1, (255, 255, 255), 1)

        print(counter)
        startTime = time.time()

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(time.time() - fullStartTime)
csvFile.close()

cap.release()
cv2.destroyAllWindows()
