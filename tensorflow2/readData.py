import glob
import time
from threading import Thread

import cv2
import csv

from tensorflow2.cnn import CnnObj
from tensorflow2.constants import EMOTIONS

import numpy as np

CSV_PATH = 'csv_data'


def write_csv(file_path, index):
    file_name = file_path.split('\\')[-1][:-4]  # not include .mp4
    csv_path = '{}/{}_{}.csv'.format(CSV_PATH, index, file_name)

    cap = cv2.VideoCapture(file_path)

    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    startTime = time.time()

    # new csv
    counter = 0
    csv_file = open(csv_path, 'w', newline='')
    csv_file_writer = csv.writer(csv_file)
    head = EMOTIONS.copy()
    head.insert(0, 'second')
    csv_file_writer.writerows([head])

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1,
                                             minNeighbors=10, minSize=(5, 5),
                                             flags=cv2.CASCADE_SCALE_IMAGE)

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
                predList = np.insert(predList, 0, 0.5 * counter)

                csv_file_writer.writerows([predList])

            startTime = time.time()

        # cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    csv_file.close()

    cap.release()
    cv2.destroyAllWindows()


def write_csv_thread(file_paths, num):
    for index, file_path in enumerate(file_paths):
        show_num = num + index
        print('{}, start...{}'.format(show_num, file_path))
        write_csv(file_path, show_num)
        print('{}, finish...{}'.format(show_num, file_path))


if __name__ == '__main__':
    file_paths = glob.glob('./origin_data/*/*/*')

    split_num = 30
    aa = int(len(file_paths) / split_num)

    for i in range(aa):
        start_num = i * split_num
        end_num = (i + 1) * split_num
        file_paths_split = file_paths[start_num:end_num]
        Thread(target=write_csv_thread, args=(file_paths_split, start_num)).start()

    print('All finish. num: {}'.format(len(file_paths)))
