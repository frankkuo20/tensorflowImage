import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow2.constants import EMOTIONS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_image_paths(img_dir):
    """
    Get all files' name under given dir img_dir
    """

    filenames = os.listdir(img_dir)
    filenames = [os.path.join(img_dir, item) for item in filenames]
    return filenames


# if __name__ == '__main__':


dataRoot = '../dataset/'

totalNum = 0

totalFiles = []
for emotion in EMOTIONS:
    filenames = get_image_paths(dataRoot+emotion)
    totalFiles.append(filenames)
    totalNum += len(filenames)

TRAIN_SEC, TEST_SEC = 0.8, 0.2

totalFilesTrain = []
totalFilesTest = []
for i, totalFile in enumerate(totalFiles):
    np.random.shuffle(totalFile)
    rate = int(len(totalFile) * TRAIN_SEC)
    filesTrain = totalFile[: rate]
    filesTest = totalFile[rate:]
    totalFilesTrain.append(filesTrain)
    totalFilesTest.append(filesTest)
    print('{}: train num: {}, test num: {}'.format(EMOTIONS[i],
                                                   len(filesTrain),
                                                   len(filesTest)))

print('=================================================')
all_train, all_test = [], []
all_train_label, all_test_label = [], []
for i, filesTrain in enumerate(totalFilesTrain):
    all_train.extend(filesTrain)
    for temp in range(len(filesTrain)):
        all_train_label.append(i)

for i, filesTest in enumerate(totalFilesTest):
    all_test.extend(filesTest)
    for temp in range(len(filesTest)):
        all_test_label.append(i)

print('Total Train Num: {}'.format(len(all_train)))
print('Total Test Num: {}'.format(len(all_test)))


print('=================================================')

def resize_img(img_path, shape):
    # resize image given by `image_path` to `shape`

    im = Image.open(img_path)
    im = im.resize(shape)
    # im = im.convert('RGB')

    # img_path = tf.image.decode_jpeg(img_path, channels=3)
    # im = tf.image.resize_images(img_path, shape)
    #
    # im = tf.cast(im, tf.uint16)

    return im

IMG_SIZE = 128  # 图像大小


def save_as_tfrecord(samples, labels, bin_path):
    #  Save images and labels as TFRecord to file: `bin_path`
    assert len(samples) == len(labels)
    writer = tf.python_io.TFRecordWriter(bin_path)
    img_label = list(zip(samples, labels))
    np.random.shuffle(img_label)
    for img, label in img_label:
        # 这里将图片的大小resize为128*128
        im = resize_img(img, (IMG_SIZE, IMG_SIZE))

        im_raw = im.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[im_raw]))
        }))
        writer.write(example.SerializeToString())
    writer.close()


save_as_tfrecord(all_test, all_test_label, "./binData/train.bin")
save_as_tfrecord(all_train, all_train_label, "./binData/test.bin")

print('bin file save finish')
