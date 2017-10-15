import glob

import matplotlib.pyplot as plt
import csv


if __name__ == '__main__':
    # filePath = glob.glob('csvData/C_075_1.csv')
    filePath = 'csvData/C_075_1.csv'

    f = open(filePath, 'r')
    secondList = []
    temp = []
    temp2 = []
    temp3 = []

    for row in csv.DictReader(f):
        secondList.append(row['second'])
        temp.append(row['neutral'])
        temp2.append(row['disgust'])
        temp3.append(row['happy'])

    f.close()

    plt.plot(secondList, temp, label='neutral', color='black')  # , 'bo-'
    plt.plot(secondList, temp2, label='disgust', color='blue')
    plt.plot(secondList, temp3, label='happy', color='orange')
    plt.legend()
    plt.xlabel('Second')
    plt.show()

