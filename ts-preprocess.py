import sys, pickle
import matplotlib.pyplot as plt
import time
import json

DISABLE_GPU = False
import os
if DISABLE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

#---------------------------------------------------------------------

_data_path = './traffic-signs-data'
_log_path = './logs'

#---------------------------------------------------------------------

# # Load pickled data
# import pickle

# training_file = 'traffic-signs-data/train.p'
# validation_file = 'traffic-signs-data/valid.p'
# testing_file = 'traffic-signs-data/test.p'

# with open(training_file, mode='rb') as f:
#     train = pickle.load(f)
# with open(validation_file, mode='rb') as f:
#     valid = pickle.load(f)
# with open(testing_file, mode='rb') as f:
#     test = pickle.load(f)

# X_train, y_train = train['features'], train['labels']
# X_valid, y_valid = valid['features'], valid['labels']
# X_test, y_test = test['features'], test['labels']

# print("--- loading: %s seconds ---" % (time.time() - start_time))

#---------------------------------------------------------------------

# # load sign names csv
# import pandas as pd

# sign_names_df = pd.read_csv("signnames.csv")
# sign_names=[]
# for i,r in sign_names_df.iterrows():
#     sign_names.append(r['SignName'])

#---------------------------------------------------------------------

# # Number of training examples
# n_train = X_train.shape[0]

# # Number of validation examples
# n_validation = X_valid.shape[0]

# # Number of testing examples.
# n_test = X_test.shape[0]

# # What's the shape of an traffic sign image?
# image_shape = X_train[0].shape

# image_w = image_shape[0]
# image_h = image_shape[1]

# # How many unique classes/labels there are in the dataset.
# n_classes = len(sign_names)

# print("Number of training examples =", n_train)
# print("Number of testing examples =", n_test)
# print("Image data shape =", image_shape)
# print("Number of classes =", n_classes)

#---------------------------------------------------------------------

import PIL.Image
from io import StringIO
from io import BytesIO
import IPython.display

def showArray(a, fmt='png', concat=False, normalized=False):
    """Display an array image RAW without any resizing. If a is an array of images and concat=True, images are concatenated horizontally."""
    if concat:
        cimages = np.empty(0)
        for i in a:
            if cimages.size==0:
                cimages=i
                continue
            cimages = np.concatenate((cimages,i),axis=1)
        a = cimages

    if normalized:
        a = a * 255

    if a.ndim==2:
        a=np.uint8(a)
    elif a.ndim==3 and a.shape[2]==1:
        a = np.uint8(a[:,:,0])
    elif a.ndim==3 and a.shape[2]==3:
        pass
    else:
        raise Exception('only 2d and 3d arrays with 3 colors supported')

    f = BytesIO()
    PIL.Image.fromarray(a).show()
    #IPython.display.display(IPython.display.Image(data=f.getvalue()))

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

def img_rotate_120(img):
    return skimage.transform.rotate(img, 120, mode='edge')

def img_rotate_240(img):
    return skimage.transform.rotate(img, 240, mode='edge')

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
    return tf.nn.dropout(input, keep_prob)

def fully_connected(input, n_output):
    input_w = int(input.get_shape()[1])
    weights = tf.get_variable('weights', shape=[input_w, n_output], initializer = tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('biases', shape=[n_output], initializer = tf.zeros_initializer())
    return tf.add(tf.matmul(input, weights), biases)

def fully_connected_relu(input, n_output):
    return tf.nn.relu(fully_connected(input, n_output))

#---------------------------------------------------------------------

def preprocess_data(X_train, X_valid, X_test):
    X_train_norm = np.array([img_equalize_adapthist(img_rgb2LNorm(xi)) for xi in X_train])[:,:,:,None]
    X_valid_norm = np.array([img_equalize_adapthist(img_rgb2LNorm(xi)) for xi in X_valid])[:,:,:,None]
    X_test_norm = np.array([img_equalize_adapthist(img_rgb2LNorm(xi)) for xi in X_test])[:,:,:,None]

    # X_train_norm = np.array([img_rgb2LNorm(xi) for xi in X_train])[:,:,:,None]
    # X_valid_norm = np.array([img_rgb2LNorm(xi) for xi in X_valid])[:,:,:,None]
    # X_test_norm = np.array([img_rgb2LNorm(xi) for xi in X_test])[:,:,:,None]

    return (X_train_norm, X_valid_norm, X_test_norm)

#---------------------------------------------------------------------

def save_data(path, prefix, X_train, X_valid, X_test, y_train, y_valid, y_test):
    print('Saving preprocessed training data with prefix:',prefix)
    with open('{}/{}_train.p'.format(path,prefix), 'wb') as handle:
        pickle.dump({'features':X_train, 'labels':y_train}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('{}/{}_valid.p'.format(path,prefix), 'wb') as handle:
        pickle.dump({'features':X_valid, 'labels':y_valid}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('{}/{}_test.p'.format(path,prefix), 'wb') as handle:
        pickle.dump({'features':X_test, 'labels':y_test}, handle, protocol=pickle.HIGHEST_PROTOCOL)

#---------------------------------------------------------------------

# n_      normalized
# n_eh_   normalized,equal_hist
# n_ec_   normalized,equal_clahe
# n_ea_   normalized,equal_adaptivehist

def load_data(path, prefix):
    with open('{}/{}_train.p'.format(path,prefix), 'rb') as handle:
        train = pickle.load(handle)
    with open('{}/{}_valid.p'.format(path,prefix), 'rb') as handle:
        valid = pickle.load(handle)
    with open('{}/{}_test.p'.format(path,prefix), 'rb') as handle:
        test = pickle.load(handle)
    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']
    return(X_train, X_valid, X_test, y_train, y_valid, y_test)

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

#---------------------------------------------------------------------


def build_network(x, enable_dropout, config):
    _=config
    k, fcr1_size, fcr2_size, cr1_d, cr2_d, cr3_d = (_['k'],_['fcr1_size'],_['fcr2_size'],_['cr1_d'],_['cr2_d'],_['cr3_d'])
    cr1_drop, cr2_drop, cr3_drop, fcr1_drop, fcr2_drop = (_['cr1_drop'], _['cr2_drop'], _['cr3_drop'], _['fcr1_drop'], _['fcr2_drop'])

    with tf.variable_scope('cr1'):
        cr1 = convolution_relu(x, K=k, S=1, D=cr1_d, padding='SAME')
        print('cr1 shape:', cr1.shape)
        cr1 = pool(cr1, 2)
        cr1 = tf.cond(enable_dropout, lambda: dropout(cr1, keep_prob=cr1_drop), lambda: cr1)

    with tf.variable_scope('cr2'):
        cr2 = convolution_relu(cr1, K=k, S=1, D=cr2_d, padding='SAME')
        print('cr2 shape:', cr2.shape)
        cr2 = pool(cr2, 2)
        cr2 = tf.cond(enable_dropout, lambda: dropout(cr2, keep_prob=cr2_drop), lambda: cr2)

    with tf.variable_scope('cr3'):
        cr3 = convolution_relu(cr2, K=k, S=1, D=cr3_d, padding='SAME')
        print('cr3 shape:', cr3.shape)
        cr3 = pool(cr3, 2)
        cr3 = tf.cond(enable_dropout, lambda: dropout(cr3, keep_prob=cr3_drop), lambda: cr3)

    flat = tf.concat((flatten(cr1),flatten(cr2),flatten(cr3)),1)
    print('flat shape:', flat.shape)

    with tf.variable_scope('fcr1'):
        fcr1 = fully_connected_relu(flat, fcr1_size)
        print('fcr1 shape:', fcr1.shape)
        fcr1 = tf.cond(enable_dropout, lambda: dropout(fcr1, keep_prob=fcr1_drop), lambda: fcr1)

    with tf.variable_scope('fcr2'):
        fcr2 = fully_connected_relu(fcr1, fcr2_size)
        print('fcr2 shape:', fcr2.shape)
        fcr2 = tf.cond(enable_dropout, lambda: dropout(fcr2, keep_prob=fcr2_drop), lambda: fcr2)

    with tf.variable_scope('fc3'):
        fc3 = fully_connected(fcr2, 43)
        print('fc3 shape:', fc3.shape)

    logits = fc3

    return(logits)

#---------------------------------------------------------------------

def flip_rotate(x, y, fn, src_class, dst_class, classes_id, classes_first_index, classes_count):
    if not classes_id[src_class]==src_class:
        raise Exception("class id doesn't match class_id position")

    new_images = []
    for i in range(classes_first_index[src_class], classes_first_index[src_class]+classes_count[src_class]):
        img = fn(x[i])
        if not len(new_images):
            new_images = img[None,:]
        else:
            new_images=np.append(new_images,img[None,:],axis=0)

    new_ids = np.full(classes_count[src_class],dst_class)

    x=np.concatenate((x,new_images),axis=0)
    y=np.concatenate((y,new_ids),axis=0)

    return(x,y)

def augment_flip_rotate(x,y):
    classes_id, classes_first_index, classes_count = np.unique(y, return_counts=True, return_index=True)
    # plt.bar(np.arange(classes_count.size), classes_count, label='n_samples')
    # plt.show()
    for i in (1,5,12,15,17):
        x,y=flip_rotate(x, y, img_flip_vertical, i, i, classes_id, classes_first_index, classes_count)
    for i in (11,12,13,15,17,18,22,26,30,35):
        x,y=flip_rotate(x, y, img_flip_horizontal, i, i, classes_id, classes_first_index, classes_count)
    for i in (12,15,17,32):
        x,y=flip_rotate(x, y, img_rotate_180, i, i, classes_id, classes_first_index, classes_count)
    for i,ii in ((19,20),(33,34),(36,37),(38,39)):
        x,y=flip_rotate(x, y, img_flip_horizontal, i, ii, classes_id, classes_first_index, classes_count)
    for i in [40]:
        x,y=flip_rotate(x, y, img_rotate_120, i, i, classes_id, classes_first_index, classes_count)
        x,y=flip_rotate(x, y, img_rotate_240, i, i, classes_id, classes_first_index, classes_count)


    # sort x and y based on y
    p = y.argsort()
    x=x[p]
    y=y[p]

    classes_id, classes_first_index, classes_count = np.unique(y, return_counts=True, return_index=True)
    # plt.bar(np.arange(classes_count.size), classes_count, label='n_samples')
    # plt.show()
    return(x,y)

#---------------------------------------------------------------------

# preprocess/save data
# X_train, X_valid, X_test = preprocess_data(X_train, X_valid, X_test)
# save_data(_data_path,'n_ea',X_train, X_valid, X_test, y_train, y_valid, y_test)
# sys.exit(0)

# X_train, X_valid, X_test, y_train, y_valid, y_test = load_data(_data_path, 'n_ea')
# print(X_train.shape,y_train.shape)
# augment_flip_rotate(X_train,y_train)

# save_data(_data_path,'n_ea_fr',X_train, X_valid, X_test, y_train, y_valid, y_test)

# sys.exit(0)

#---------------------------------------------------------------------

from configs import configs as configs

BATCH_SIZE = 256

#---------------------------------------------------------------------

_c=None
if len(sys.argv)==2:
    _c=configs[int(sys.argv[1])]
else:
    _c=configs[-1]
print(_c)

X_train, X_valid, X_test, y_train, y_valid, y_test = load_data(_data_path, _c['prefix'])

from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)

image_shape = X_train[0].shape
image_w = image_shape[0]
image_h = image_shape[1]
classes_id, classes_first_index, classes_count = np.unique(y_train, return_counts=True, return_index=True)
n_classes=classes_id.size

x = tf.placeholder(tf.float32, (None, image_w, image_h, 1))
y = tf.placeholder(tf.int32, (None))
enable_dropout = tf.placeholder(tf.bool)
y_one_hot = tf.one_hot(y, n_classes)

logits = build_network(x,enable_dropout, _c)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = _c['learn_rate'])
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
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, enable_dropout: False})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

validation_log = []
train_time = 0
test_accuracy=0

start_time = time.time()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(_c['epochs']):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, enable_dropout: True})

        validation_accuracy = evaluate(X_valid, y_valid)
        validation_log.append(validation_accuracy)

        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    train_time = time.time() - start_time
    print("--- training: %s seconds ---" % (time.time() - start_time))

    saver.save(sess, './train-sessions/session')
    print("Model saved")

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./train-sessions'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

json_data = {}
json_data['config']=_c
json_data['validation_accuracy']=validation_log
json_data['test_accuracy']=test_accuracy
json_data['train_time']=train_time
with open('{}/{}.json'.format(_log_path,_c['name']), 'w') as outfile:
    json.dump(json_data, outfile, indent=4)
