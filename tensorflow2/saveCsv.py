import csv
import glob
import numpy as np

from tensorflow2.cnn import CnnObj
from PIL import Image

from tensorflow2.constants import EMOTIONS


def readImg(filePath):
    image = Image.open(filePath)
    cnnObj = CnnObj()
    predList = cnnObj.getPredList(image)
    return predList


if __name__ == '__main__':
    filePaths = glob.glob('imageData/C_075_1/*')
    csv_path = 'csvData/C_075_1.csv'

    # new csv
    counter = 0
    csv_file = open(csv_path, 'w', newline='')
    csv_file_writer = csv.writer(csv_file)
    head = EMOTIONS.copy()
    head.insert(0, 'second')
    csv_file_writer.writerows([head])

    counter = 0
    for filePath in filePaths:
        counter += 1
        predList = readImg(filePath)
        predList = np.insert(predList, 0, 0.5 * counter)
        csv_file_writer.writerows([predList])

    csv_file.close()
