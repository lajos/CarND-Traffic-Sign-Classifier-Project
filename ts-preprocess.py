import time
start_time = time.time()

DISABLE_GPU = False
import os
if DISABLE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tensorflow.examples.tutorials.mnist import input_data

#---------------------------------------------------------------------

# Load pickled data
import pickle

training_file = 'traffic-signs-data/train.p'
validation_file = 'traffic-signs-data/valid.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

print("--- loading: %s seconds ---" % (time.time() - start_time))

#---------------------------------------------------------------------

# load sing names csv
import pandas as pd

sign_names_df = pd.read_csv("signnames.csv")
sign_names=[]
for i,r in sign_names_df.iterrows():
    sign_names.append(r['SignName'])

#---------------------------------------------------------------------

# Number of training examples
n_train = X_train.shape[0]

# Number of validation examples
n_validation = X_valid.shape[0]

# Number of testing examples.
n_test = X_test.shape[0]

# What's the shape of an traffic sign image?
image_shape = X_train[0].shape

image_w = image_shape[0]
image_h = image_shape[1]

# How many unique classes/labels there are in the dataset.
n_classes = len(sign_names)

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

#---------------------------------------------------------------------

import numpy as np
import skimage.transform
import random
import cv2

def rgb2LNorm(img):
    #print(img)
    labImg = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    labImg = labImg[:,:,0]
    return labImg/255.

def flip_vertical(img):
    return np.flipud(img)

def flip_horizontal(img):
    return np.fliplr(img)

def rotate_180(img):
    return flip_vertical(flip_horizontal(img))

def rotate_random(img, degrees=5):
    """ random rotation between degrees cw and degrees ccw """
    return skimage.transform.rotate(img, random.uniform(-degrees,degrees), mode='edge')

def project_random(img, max_distance=5):
    """ projection transformation, moving each corner max_distance pixels from original """
    w,h=img.shape
    matrix = np.array(((0,0),(0,h),(w,h),(w,0)))
    projection = skimage.transform.ProjectiveTransform()
    projection.estimate(matrix+(2.*np.random.rand(matrix.size).reshape(matrix.shape)-1)*max_distance,matrix)
    img = skimage.transform.warp(img, projection, mode='edge')
    return(img)

#---------------------------------------------------------------------

import numpy as np
import skimage.transform
import skimage.morphology
import skimage.filters.rank
import skimage.exposure
import random
import warnings

def img_rgb2LNorm(img):
    labImg = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    labImg = labImg[:,:,0]
    return labImg/255.

def img_flip_vertical(img):
    return np.flipud(img)

def img_flip_horizontal(img):
    return np.fliplr(img)

def img_rotate_180(img):
    return flip_vertical(flip_horizontal(img))

def img_rotate_random(img, degrees=5):
    """ random rotation between degrees cw and degrees ccw """
    return skimage.transform.rotate(img, random.uniform(-degrees,degrees), mode='edge')

def img_project_random(img, max_distance=5):
    """ projection transformation, moving each corner max_distance pixels from original """
    w,h=img.shape
    matrix = np.array(((0,0),(0,h),(w,h),(w,0)))
    projection = skimage.transform.ProjectiveTransform()
    projection.estimate(matrix+(2.*np.random.rand(matrix.size).reshape(matrix.shape)-1)*max_distance,matrix)
    img = skimage.transform.warp(img, projection, mode='edge')
    return(img)

def img_equalize_hist(img):
    return cv2.equalizeHist((img*255).astype(np.uint8))/255.

def img_equalize_clahe(img, clipLimit=4, tileSize=14):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tileSize,tileSize))
    return clahe.apply((img*255).astype(np.uint8))/255.

def img_equalize_adapthist(img):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = skimage.exposure.equalize_adapthist(img)
    return(img)

def img_median(img, size=5):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = 1-skimage.filters.rank.median(img, skimage.morphology.disk(size))
    return(img)

def img_threshold_local(img, size=5):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = skimage.filters.threshold_local(img, size, method='gaussian')
    return(img)

def img_random_noise(img):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = skimage.util.random_noise(img)
    return(img)

#---------------------------------------------------------------------

import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np

def convolution_relu(input, K=5, S=1, D=16, padding='SAME'):
    # W1 x H1 x D1 input
    # K number of filters
    # F spatial intent (size)
    # S stride
    # P padding
    # W2 = (W1−F+2P)/S+1
    # H2 = (H1−F+2P)/S+1
    # D2 = K

    input_w = input.shape[1]
    input_h = input.shape[2]
    input_d = 1
    if len(input.shape)==4:
        input_d = input.shape[3]

    weights = tf.get_variable('weights', shape=[K,K,input_d,D], initializer = tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('biases', shape=[D], initializer = tf.zeros_initializer())

    output = tf.nn.conv2d(input, weights, strides=[1,S,S,1], padding=padding)
    output = tf.nn.bias_add(output, biases)

    output = tf.nn.relu(output)
    return output

def pool(input, k_size):
    return tf.nn.max_pool(input, [1, k_size, k_size, 1], [1, k_size, k_size, 1], padding='SAME')

def dropout(input, keep_prob=0.5):
    return tf.dropout(input, keep_prob)

def fully_connected(input, n_output):
    input_w = int(input.get_shape()[1])
    weights = tf.get_variable('weights', shape=[input_w, n_output], initializer = tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('biases', shape=[n_output], initializer = tf.zeros_initializer())
    return tf.add(tf.matmul(input, weights), biases)

def fully_connected_relu(input, n_output):
    return tf.nn.relu(fully_connected(input, n_output))

#---------------------------------------------------------------------


def lenet_model(x):
    with tf.variable_scope('cr1'):
        cr1 = convolution_relu(x, K=5, S=1, D=6, padding='SAME')
        print('cr1 shape:', cr1.shape)
        cr1 = pool(cr1, 2)

    with tf.variable_scope('cr2'):
        cr2 = convolution_relu(cr1, K=5, S=1, D=16, padding='SAME')
        print('cr2 shape:', cr2.shape)
        cr2 = pool(cr2, 2)

    flat = flatten(cr2)
    print('flat shape:', flat.shape)

    with tf.variable_scope('fcr1'):
        fcr1 = fully_connected_relu(flat, 120)
        print('fcr1 shape:', fcr1.shape)

    with tf.variable_scope('fcr2'):
        fcr2 = fully_connected_relu(fcr1, 84)
        print('fcr2 shape:', fcr2.shape)

    with tf.variable_scope('fc3'):
        fc3 = fully_connected(fcr2, 43)
        print('fc3 shape:', fc3.shape)

    logits = fc3

    return(fc3)


def training_model(x):
    with tf.variable_scope('cr1'):
        cr1 = convolution_relu(x, K=5, S=1, D=32, padding='SAME')
        print('cr1 shape:', cr1.shape)
        cr1 = pool(cr1, 2)

    with tf.variable_scope('cr2'):
        cr2 = convolution_relu(cr1, K=5, S=1, D=64, padding='SAME')
        print('cr2 shape:', cr2.shape)
        cr2 = pool(cr2, 2)

    with tf.variable_scope('cr3'):
        cr3 = convolution_relu(cr2, K=5, S=1, D=128, padding='SAME')
        print('cr3 shape:', cr3.shape)
        cr3 = pool(cr3, 2)


    flat = tf.concat((flatten(cr1),flatten(cr2),flatten(cr3)),1)
    print('flat shape:', flat.shape)

    with tf.variable_scope('fcr1'):
        fcr1 = fully_connected_relu(flat, 2048)
        print('fcr1 shape:', fcr1.shape)

    with tf.variable_scope('fcr2'):
        fcr2 = fully_connected_relu(fcr1, 512)
        print('fcr2 shape:', fcr2.shape)

    with tf.variable_scope('fc3'):
        fc3 = fully_connected(fcr2, 43)
        print('fc3 shape:', fc3.shape)

    logits = fc3

    return(fc3)

X_train_norm = np.array([img_equalize_adapthist(img_rgb2LNorm(xi)) for xi in X_train])
X_valid_norm = np.array([img_equalize_adapthist(img_rgb2LNorm(xi)) for xi in X_valid])
X_test_norm = np.array([img_equalize_adapthist(img_rgb2LNorm(xi)) for xi in X_test])

print("--- data norm: %s seconds ---" % (time.time() - start_time))

X_train = X_train_norm[:,:,:,None]
X_validation = X_valid_norm[:,:,:,None]
X_test = X_test_norm[:,:,:,None]

y_validation = y_valid

from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)

x = tf.placeholder(tf.float32, (None, image_w, image_h, 1))
y = tf.placeholder(tf.int32, (None))
y_one_hot = tf.one_hot(y, n_classes)

EPOCHS = 20
BATCH_SIZE = 256
rate = 0.0005

logits = training_model(x)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_one_hot, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    print("--- training: %s seconds ---" % (time.time() - start_time))

    saver.save(sess, './lenet')
    print("Model saved")

print("--- saving: %s seconds ---" % (time.time() - start_time))

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

print("--- total: %s seconds ---" % (time.time() - start_time))
