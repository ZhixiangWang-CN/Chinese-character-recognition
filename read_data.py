import os
import numpy as np
import cv2
import random
import tensorflow as tf

path = './data/'
file = 'x.txt'
zhongshu = 12180


def get_filelist(dir):
    Filelist = []
    Y_list = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            # print(files)
            # 文件名列表，包含完整路径
            # print(home)
            new_name = home + '/' + filename
            Filelist.append(new_name)
            # h_name = home.split('\\')
            # new_name =
            # Y_list.append(h_name[-1])
            # # 文件名列表，只包含文件名
            # Filelist.append( filename)

    return Filelist, Y_list


def write_txt(Filelist):
    n = len(Filelist)
    fx = open("x.txt", "w")
    for i in range(n):
        fx.write(Filelist[i] + '\n')
        # print("saving",i)
    fx.close()
    return None


def read_txt(file):
    with open(file, 'r') as f:
        content = f.readlines()
    return content


def read_image(path):
    name = path[:-1]
    image = cv2.imread(name)
    image = cv2.resize(image, (64, 64))
    h_name = name.split('/')
    return image, h_name[-2]


# def to_one_hot(y, n_class):
#     NUM_CLASSES = n_class
#     labels = y
#     batch_size = tf.size(labels)  # get size of labels : 4
#     labels = tf.expand_dims(labels, 1)  # 增加一个维度

#     indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)  # 生成索引
#     print("indices",indices)
#     labels = tf.cast(labels,dtype=tf.int32)
#     print("labels",labels)

#     concated,labels = tf.concat([indices, labels], 1)  # 作为拼接
#     onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, NUM_CLASSES]), 1.0, 0.0)
#     return onehot_labels
def to_one_hot(y, n_class):
    #     y = np.ndarray(y)
    #     y = int(y)
    #     print(y)
    arr = []
    for k in y:
        p = int(k)
        #         print(p.type)
        arr.append(p)
    num_classes = n_class
    # 需要转换的整数

    #     print(arr)
    # 将整数转为一个10位的one hot编码
    k = np.eye(n_class)[arr]
    #     print(k)
    return k


def read_image_batch(batch_size):
    X_image = []
    y = []
    content = read_txt(file)
    for i in range(batch_size):
        ran = random.randint(0, zhongshu)

        img, h_name = read_image(content[ran])
        X_image.append(img)
        y.append(h_name)
    X_image = np.array(X_image)
    print(" X_image", X_image.shape)
    return X_image, y


if __name__ == "__main__":
    content = read_txt(file)
    X_image, y = read_image_batch(16)
    yy = to_one_hot(y, 65)
#     print(yy)



