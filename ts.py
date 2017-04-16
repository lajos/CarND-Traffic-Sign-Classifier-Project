import sys, pickle
import matplotlib.pyplot as plt
import time
import json
import cv2

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
    return img_flip_vertical(img_flip_horizontal(img))

def img_rotate_random(img, degrees=5):
    """ random rotation between degrees cw and degrees ccw """
    return skimage.transform.rotate(img, random.uniform(-degrees,degrees), mode='edge')

def img_scale_random(img, scale_diff=0.2):
    transform = skimage.transform.AffineTransform(scale=(random.uniform(1.0-scale_diff, 1.0+scale_diff),random.uniform(1.0-scale_diff, 1.0+scale_diff)))
    img = skimage.transform.warp(img, transform.inverse, mode='edge')
    return(img)

def img_rotate_120(img):
    return skimage.transform.rotate(img, 120, mode='edge')

def img_rotate_240(img):
    return skimage.transform.rotate(img, 240, mode='edge')

def img_project_random(img, max_distance=5):
    """ projection transformation, moving each corner max_distance pixels from original """
    w=img.shape[0]
    h=img.shape[1]
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

def img_median(img, size=1):
    img=img.reshape(img.shape[0],img.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = 1-skimage.filters.rank.median(img, skimage.morphology.disk(size))
    return(img[:,:,None])

def img_threshold_local(img, size=1):
    img=img.reshape(img.shape[0],img.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = skimage.filters.threshold_local(img, size, method='gaussian')
    return(img[:,:,None])

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

def preprocess_data(X_train, X_valid, X_test,t='n'):
    X_train_norm = None
    X_valid_norm = None
    X_test_norm = None

    if t=='n':
        X_train_norm = np.array([img_rgb2LNorm(xi) for xi in X_train])[:,:,:,None]
        X_valid_norm = np.array([img_rgb2LNorm(xi) for xi in X_valid])[:,:,:,None]
        X_test_norm = np.array([img_rgb2LNorm(xi) for xi in X_test])[:,:,:,None]
    elif t=='n_eh':
        X_train_norm = np.array([img_equalize_hist(img_rgb2LNorm(xi)) for xi in X_train])[:,:,:,None]
        X_valid_norm = np.array([img_equalize_hist(img_rgb2LNorm(xi)) for xi in X_valid])[:,:,:,None]
        X_test_norm = np.array([img_equalize_hist(img_rgb2LNorm(xi)) for xi in X_test])[:,:,:,None]
    elif t=='n_ec':
        X_train_norm = np.array([img_equalize_clahe(img_rgb2LNorm(xi)) for xi in X_train])[:,:,:,None]
        X_valid_norm = np.array([img_equalize_clahe(img_rgb2LNorm(xi)) for xi in X_valid])[:,:,:,None]
        X_test_norm = np.array([img_equalize_clahe(img_rgb2LNorm(xi)) for xi in X_test])[:,:,:,None]
    elif t=='n_ea':
        X_train_norm = np.array([img_equalize_adapthist(img_rgb2LNorm(xi)) for xi in X_train])[:,:,:,None]
        X_valid_norm = np.array([img_equalize_adapthist(img_rgb2LNorm(xi)) for xi in X_valid])[:,:,:,None]
        X_test_norm = np.array([img_equalize_adapthist(img_rgb2LNorm(xi)) for xi in X_test])[:,:,:,None]


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
    with open('{}/{}_train.p'.format(path, prefix), 'rb') as handle:
        train = pickle.load(handle)
    with open('{}/{}_valid.p'.format(path, prefix), 'rb') as handle:
        valid = pickle.load(handle)
    with open('{}/{}_test.p'.format(path, prefix), 'rb') as handle:
        test = pickle.load(handle)
    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']
    return(X_train, X_valid, X_test, y_train, y_valid, y_test)

#---------------------------------------------------------------------

def load_www_data(path, prefix):
    with open('{}/{}_www.p'.format(path, prefix), 'rb') as handle:
        www = pickle.load(handle)
    X_www, y_www = www['features'], www['labels']
    return(X_www, y_www)

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

def build_network_parallel(x, enable_dropout, config):
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

    with tf.variable_scope('cr1a'):
        cr1a = convolution_relu(x, K=5, S=1, D=cr1_d, padding='SAME')
        print('cr1a shape:', cr1a.shape)
        cr1a = pool(cr1a, 2)
        cr1a = tf.cond(enable_dropout, lambda: dropout(cr1a, keep_prob=cr1_drop), lambda: cr1a)

    with tf.variable_scope('cr2a'):
        cr2a = convolution_relu(cr1a, K=5, S=1, D=cr2_d, padding='SAME')
        print('cr2a shape:', cr2a.shape)
        cr2a = pool(cr2a, 2)
        cr2a = tf.cond(enable_dropout, lambda: dropout(cr2a, keep_prob=cr2_drop), lambda: cr2a)

    flat = tf.concat((flatten(cr1),flatten(cr2),flatten(cr3), flatten(cr1a),flatten(cr2a)),1)
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

def get_scope_variable(scope_name, var_name):
    with tf.variable_scope(scope_name, reuse=True):
        v = tf.get_variable(var_name)
    return v

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
import random

def augment_filter_class(x, y, c_id, c_index, c_count, aug_count):
    print("filter augment:",c_id, c_index, c_count, aug_count)

    if aug_count==0:
        return x,y

    new_images = []
    for i in range(aug_count):
        src_index = random.randrange(c_index, c_index+c_count)

        img = x[src_index]

        img = img_project_random(img, max_distance=5)               # normal (1,7)  large (1,11)
        img = img_rotate_random(img, degrees=5)                    # normal (0,10) large (0,20)
        img = img_scale_random(img, scale_diff=0.1)                # normal (0,0.2) large (0,0.4)

        # if (random.random()<0.1):
        #     img = img_median(img, size=1)
        # if (random.random()<0.1):
        #     img = img_threshold_local(img, size=random.randrange(1,4,2))

        # if (random.random()<0.2):                                                    # normal 0.1 large 0.1
        #     img = img_random_noise(img)

        if not len(new_images):
            new_images = img[None,:]
        else:
            new_images=np.append(new_images,img[None,:],axis=0)

    new_ids = np.full(aug_count,c_id)

    # showArray(new_images[0])
    # print('new_id',new_ids[0])
    # print(new_images.shape)
    # print(new_ids.shape)

    x=np.concatenate((x,new_images),axis=0)
    y=np.concatenate((y,new_ids),axis=0)

    return x,y

def augment_filter(x,y,max_samples=-1):
    """if max_sample==-1, augment to largest class count"""
    classes_id, classes_first_index, classes_count = np.unique(y, return_counts=True, return_index=True)
    if max_samples==-1:
        max_samples = np.max(classes_count)
    for c_id, c_index, c_count in zip (classes_id, classes_first_index, classes_count):
        aug_count = max_samples - c_count
        x,y=augment_filter_class(x, y, c_id, c_index, c_count, aug_count)

    classes_id, classes_first_index, classes_count = np.unique(y, return_counts=True, return_index=True)
    # plt.bar(np.arange(classes_count.size), classes_count, label='n_samples')
    # plt.show()
    return(x,y)


#---------------------------------------------------------------------

def get_y_imbalance_weights(y):
    classes_id, classes_first_index, classes_count = np.unique(y, return_counts=True, return_index=True)
    print('min samples per class: ',np.min(classes_count))
    print('max samples per class: ',np.max(classes_count))
    w = np.min(classes_count)/classes_count
    print(classes_count)
    print(w)
    return w


#---------------------------------------------------------------------

start_time = time.time()

# preprocess/save data (convert to L(ab), equalize)
#
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

# pre_type = 'n_ea'
# X_train, X_valid, X_test = preprocess_data(X_train, X_valid, X_test, t=pre_type)
# save_data(_data_path,pre_type,X_train, X_valid, X_test, y_train, y_valid, y_test)

#flip rotate augment
#
# X_train, X_valid, X_test, y_train, y_valid, y_test = load_data(_data_path, 'n_ea')
# print(X_train.shape,y_train.shape)
# X_train,y_train = augment_flip_rotate(X_train,y_train)
# save_data(_data_path,'n_ea_fr',X_train, X_valid, X_test, y_train, y_valid, y_test)

#filter augment
#
# X_train, X_valid, X_test, y_train, y_valid, y_test = load_data(_data_path, 'n_ea_fr')
# print(X_train.shape,y_train.shape)
# X_train,y_train = augment_filter(X_train,y_train)
# print(X_train.shape,y_train.shape)
# save_data(_data_path,'n_ea_fr_fall',X_train, X_valid, X_test, y_train, y_valid, y_test)

# print("--- processing time: %s seconds ---" % (time.time() - start_time))

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

_c['prefix']='n_ea'

X_train, X_valid, X_test, y_train, y_valid, y_test = load_data(_data_path, _c['prefix'])
X_www, y_www = load_www_data(_data_path, 'n_ea')

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

cr1_weights = get_scope_variable('cr1', 'weights')
cr2_weights = get_scope_variable('cr2', 'weights')
cr3_weights = get_scope_variable('cr3', 'weights')
fcr1_weights = get_scope_variable('fcr1', 'weights')
fcr2_weights = get_scope_variable('fcr2', 'weights')
fc3_weights = get_scope_variable('fc3', 'weights')

#l2_weights = [cr1_weights, cr2_weights, cr3_weights, fcr1_weights, fcr2_weights, fc3_weights]

#l2_weights = [cr1_weights, cr2_weights, cr3_weights]

l2_weights = [fcr1_weights, fcr2_weights, fc3_weights]

#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot, logits=logits)
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_one_hot, logits=logits)
#cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets=y_one_hot, logits=logits, pos_weight=get_y_imbalance_weights(y_train))

loss_operation = tf.reduce_mean(cross_entropy)

for w in l2_weights:
    loss_operation = loss_operation + 0.0001*tf.nn.l2_loss(w)

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
        correct, accuracy = sess.run((correct_prediction, accuracy_operation), feed_dict={x: batch_x, y: batch_y, enable_dropout: False})
        # print(correct)
        # for i in range(correct.size):
        #     if not correct[i]:
        #         print('missed:',i+1)
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

#---- www test -------
# with tf.Session() as sess:
#     saver.restore(sess, tf.train.latest_checkpoint('./train-sessions'))
#     test_accuracy = evaluate(X_www, y_www)
#     print("www Test Accuracy = {:.3f}".format(test_accuracy))
# sys.exit(0)
#---------------------

validation_log = []
test_log = []
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

        test_accuracy = evaluate(X_test, y_test)
        test_log.append(test_accuracy)

        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print("Test Accuracy = {:.3f}".format(test_accuracy))
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
json_data['test_accuracy']=test_log
json_data['test_accuracy_final']=test_accuracy
json_data['train_time']=train_time
with open('{}/{}.json'.format(_log_path,_c['name']), 'w') as outfile:
    json.dump(json_data, outfile, indent=4)
