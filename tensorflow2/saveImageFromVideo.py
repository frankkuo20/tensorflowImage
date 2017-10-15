import glob
import time
from threading import Thread

import cv2
import os

SAVE_PATH = 'imageData'


def write_image(file_path, file_name):
    file_name2 = file_path.split('\\')[-1][:-4]  # not include .mp4
    folder_name = '{}/{}_{}/'.format(SAVE_PATH, file_name, file_name2)
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    cap = cv2.VideoCapture(file_path)

    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    startTime = time.time()

    # new csv
    counter = 0

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
                cv2.imshow('frame', image)
                image_path = '{}{}_{}_{}.jpg'.format(folder_name, file_name, file_name2, counter)
                cv2.imwrite(image_path, image)

            startTime = time.time()

        # cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def write_img_thread(file_paths, num):
    for index, file_path in enumerate(file_paths):
        folderName = file_path.split('\\')[-3]
        file_name = None
        show_num = num + index
        if folderName == '老師專題':
            file_name = 'A'
        elif folderName == '資管一A':
            file_name = 'B'
        elif folderName == '資管二C':
            file_name = 'C'
        print('{}, start...{}'.format(show_num, file_path))
        write_image(file_path, file_name)
        print('{}, finish...{}'.format(show_num, file_path))


def get_fileNma(folderName):
    if folderName == '老師專題':
        file_name = 'A'
    elif folderName == '資管一A':
        file_name = 'B'
    elif folderName == '資管二C':
        file_name = 'C'
    return file_name


if __name__ == '__main__':
    file_paths = glob.glob('./origin_data/*/*/*')

    # write_img_thread(file_paths, 0)

    total = 180
    split_num = 30
    aa = int(len(file_paths) / split_num)

    for i in range(aa):
        start_num = i * split_num
        end_num = (i + 1) * split_num

        file_paths_split = file_paths[start_num:end_num]
        Thread(target=write_img_thread, args=(file_paths_split, start_num)).start()

    print('All finish. num: {}'.format(len(file_paths)))


