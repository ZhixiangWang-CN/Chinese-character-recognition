# Copyright (c) 2018 by huyz. All Rights Reserved.

import tensorflow as tf
import read_data
import os
from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

learning_rate = 0.0001
num_epoches = 200
batch_size = 16

X = tf.placeholder(tf.float32, shape=[16, 64, 64, 3])
X_img = tf.reshape(X, shape=[16, 64, 64, 3])
Y = tf.placeholder(tf.float32, shape=[16, 65])


def VGG16(x):
    ########## first conv ##########
    ##### conv1_1 #####
    print('x.shape', x)
    conv1 = tf.layers.conv2d(inputs=x, filters=64, kernel_size=3, strides=1, padding="SAME")
    conv1 = tf.nn.relu(conv1)

    ##### conv1_2 #####
    conv1 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=3, strides=1, padding="SAME")
    conv1 = tf.nn.relu(conv1)

    ###### max pool ######
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    ########## second conv ##########
    ##### conv2_1 #####
    conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=3, strides=1, padding="SAME")
    conv2 = tf.nn.relu(conv2)

    ##### conv2_2 #####
    conv2 = tf.layers.conv2d(inputs=conv2, filters=128, kernel_size=3, strides=1, padding="SAME")
    conv2 = tf.nn.relu(conv2)

    ##### max pool #####
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    ########## third conv ##########
    ##### conv3_1 #####
    conv3 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=3, strides=1, padding="SAME")
    conv3 = tf.nn.relu(conv3)

    ##### conv3_2 #####
    conv3 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=3, strides=1, padding="SAME")
    conv3 = tf.nn.relu(conv3)

    ##### conv3_3 #####
    conv3 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=1, strides=1, padding="SAME")
    conv3 = tf.nn.relu(conv3)

    ##### max pool #####
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    ########## fourth conv ##########
    ##### conv4_1 #####
    conv4 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=3, strides=1, padding="SAME")
    conv4 = tf.nn.relu(conv4)

    ##### conv4_2 #####
    conv4 = tf.layers.conv2d(inputs=conv4, filters=512, kernel_size=3, strides=1, padding="SAME")
    conv4 = tf.nn.relu(conv4)

    ##### conv4_3 #####
    conv4 = tf.layers.conv2d(inputs=conv4, filters=512, kernel_size=1, strides=1, padding="SAME")
    conv4 = tf.nn.relu(conv4)

    ##### max pool #####
    pool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    ########## fifth conv ##########
    ##### conv5_1 #####
    conv5 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=3, strides=1, padding="SAME")
    conv5 = tf.nn.relu(conv5)

    ##### conv5_2 #####
    conv5 = tf.layers.conv2d(inputs=conv5, filters=512, kernel_size=3, strides=1, padding="SAME")
    conv5 = tf.nn.relu(conv5)

    ##### conv5_3 #####
    conv5 = tf.layers.conv2d(inputs=conv5, filters=512, kernel_size=1, strides=1, padding="SAME")
    conv5 = tf.nn.relu(conv5)

    ##### max pool #####
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    print('pool5.shape', pool5.shape)

    flatten = tf.reshape(pool5, shape=[-1, 2 * 2 * 512])
    print('flatten.shape', flatten.shape)
    ########## fc1 ##########
    fc1 = tf.layers.dense(inputs=flatten, units=4096)
    fc1 = tf.nn.relu(fc1)
    print('fc1.shape', fc1.shape)

    ########## fc2 ##########
    fc2 = tf.layers.dense(inputs=fc1, units=4096)
    fc2 = tf.nn.relu(fc2)
    print('fc2.shape', fc2.shape)

    ########## fc3 ##########
    fc3 = tf.layers.dense(inputs=fc2, units=65)
    print('fc3.shape', fc3.shape)

    ########## softmax ##########
    logit = tf.nn.softmax(fc3)
    print('logit.shape', logit.shape)
    return logit


########## define model, loss and optimizer ##########
pred = VGG16(X_img)
print('pred.shape', pred.shape)
print('Y.shape', Y.shape)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
cost_summary = tf.summary.scalar("cost", cost)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

########## accuracy ##########
is_correction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correction, tf.float32))
accuracy_summary = tf.summary.scalar("accuracy", accuracy)

########## train and evaluation ##########

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
config = tf.ConfigProto(gpu_options=gpu_options)
sess = tf.Session(config=config)

saver = tf.train.Saver()

init = tf.global_variables_initializer()
sess.run(init)

'''
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs/VGG")
writer.add_graph(sess.graph)
'''

merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs/VGG")
writer.add_graph(sess.graph)

print("Learning start...")

for epoch in range(num_epoches):
    avg_acc = 0
    avg_cost = 0
    # num_batches = int(mnist.train.num_examples / batch_size)
    num_batches = 4000
    batch_x, batch_y = read_data.read_image_batch(batch_size)

    batch_y = read_data.to_one_hot(batch_y, 65)
    # batch_x, batch_y = mnist.train.next_batch(batch_size)
    feed_dict = {X: batch_x, Y: batch_y}

    for i in range(num_batches):
        if i % 100 == 0:
            print("i = ", i)
        #         summary, _, c, a =sess.run([opt,loss],feed_dict={net_input:input_image_batch,net_output:output_image_batch})
        summary, _, c, a = sess.run([merged_summary, optimizer, cost, accuracy], feed_dict=feed_dict)
        writer.add_summary(summary, global_step=i)
        avg_acc += a / num_batches
        avg_cost += c / num_batches

    print("Epoch: {}\tLoss:{:.9f}\tAccuarcy: {:.2%}".format(epoch + 1, avg_cost, avg_acc))
print("Learning finished!")
path_save = './ckpt_examples'
is_exist = os.path.exists(path_save)
if not is_exist:
    os.makedirs(path_save)
saver.save(sess, path_save + "VGG.ckpt")
print("Model saved!")

with tf.Session() as sess:
    model_file = tf.train.latest_checkpoint("./")
    saver.restore(sess, model_file)
    batch_x, batch_y = read_data.read_image_batch(batch_size)

    batch_y = read_data.to_one_hot(batch_y, 65)
    # batch_x, batch_y = mnist.train.next_batch(batch_size)
    feed_dict = {X: batch_x, Y: batch_y}

    accuracy = sess.run(accuracy, feed_dict=feed_dict)
    print("Accuracy on test: {:.3%}".format(accuracy))
