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
        img = exposure.equalize_adapthist(img)
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
    tf.nn.relu(fully_connected(input, n_output))

#---------------------------------------------------------------------

def training_model(x):
    with tf.variable_scope('cr1'):
        o = convolution_relu(x, K=5, S=1, D=6, padding='SAME')
        print('c1 shape:', o.shape)
        o = pool(o, 2)

    with tf.variable_scope('cr2'):
        o = convolution_relu(o, K=5, S=1, D=16, padding='SAME')
        print('c2 shape:', o.shape)
        o = pool(o, 2)

    o = flatten(o)

    with tf.variable_scope('fcr1'):
        o = fully_connected_relu(o, 120)
        print('fc1 shape:', o.shape)

    with tf.variable_scope('fcr2'):
        o = fully_connected_relu(o, 84)
        print('fc2 shape:', o.shape)

    with tf.variable_scope('fc3'):
        o = fully_connected(o, 43)
        print('fc3 shape:', o.shape)

    return(o)

#X_train_norm = np.array([img_rgb2LNorm(xi) for xi in X_train])
#X_valid_norm = np.array(img_[rgb2LNorm(xi) for xi in X_train])
#X_test_norm = np.array(img_[rgb2LNorm(xi) for xi in X_train])
#y_train_onehot = tf.one_hot(y_train, 10)

x = tf.placeholder(tf.float32, (None, image_w, image_h, 1))
y = tf.placeholder(tf.int32, (None))
y_one_hot = tf.one_hot(y, n_classes)

rate = 0.001

logits = training_model(x)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_train_onehot, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_one_hot, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

