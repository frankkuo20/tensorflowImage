import matplotlib.pyplot as plt
import csv

# from tensorflow2.constants import EMOTIONS
# ['neutral', 'disgust', 'happy']


def readCsv(fileName):
    f = open(fileName, 'r')
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

    return secondList, temp, temp2, temp3

secondList, temp, temp2, temp3 = readCsv('075_11.csv')

fig, (a1, a2) = plt.subplots(1, 2)
a1.plot(secondList, temp, label='neutral', color='black')  # , 'bo-'
a1.plot(secondList, temp2, label='disgust', color='blue')
a1.plot(secondList, temp3, label='happy', color='orange')
a1.legend()
a1.set_xlabel('Second')
a2.set_title('happy')

secondList, temp, temp2, temp3 = readCsv('075_22.csv')

a2.plot(secondList, temp, label='neutral', color='black')  # , 'bo-'
a2.plot(secondList, temp2, label='disgust', color='blue')
a2.plot(secondList, temp3, label='happy', color='orange')
a2.legend()
a2.set_xlabel('Second')
a1.set_title('disgust')


# plt.xlabel('Second')


plt.show()





