import matplotlib.pyplot as plt
import csv

# from tensorflow2.constants import EMOTIONS
# ['neutral', 'disgust', 'happy']

f = open('075_1.csv', 'r')
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



