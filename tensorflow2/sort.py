# source: http://www.paulvangent.com/2016/04/01/emotion-recognition-with-python-opencv-and-a-face-dataset/
import glob
import shutil
import os

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]  # Define emotion order

participants = glob.glob(
    "../ck+/Emotion_labels/Emotion/*/*/*")  # Returns a list of all folders with participant numbers

IMAGE_PATH = '../ck+/extended-cohn-kanade-images/cohn-kanade-images/{}/{}/*'
if not os.path.exists('../sorted_set'):
    os.mkdir('../sorted_set')

for x in participants:
    file = open(x)
    emotionIndex = int(float(file.readline()))
    filePath = x.split('\\')
    folder = filePath[-3]
    folder2 = filePath[-2]

    imagePath = IMAGE_PATH.format(folder, folder2)

    images = glob.glob(imagePath)

    target_path = '../sorted_set/neutral/'
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    shutil.copy(images[0], target_path)

    currentEmotion = emotions[emotionIndex]
    target_path2 = '../sorted_set/{}/'.format(currentEmotion)
    if not os.path.exists(target_path2):
        os.mkdir(target_path2)

    [shutil.copy(i, target_path2) for i in images[-3:]]
    file.close()


resultPath = '../sorted_set/{}/*'
for emotion in emotions:
    emotionPath = resultPath.format(emotion)

    print('{} nums: {}'.format(emotion.ljust(8), len(glob.glob(emotionPath))))


