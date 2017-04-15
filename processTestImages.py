import cv2
import pandas as pd
import numpy as np
import skimage.transform
import skimage.exposure
import warnings
import pickle

_test_images_folder = 'test-images'
_data_folder = 'traffic-signs-data'

def img_resize32_cv2(img):
    img = cv2.resize(img, (32,32), cv2.INTER_LANCZOS4)
    return img

def img_rgb2LNorm(img):
    labImg = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    labImg = labImg[:,:,0]
    # labImg = cv2.fastNlMeansDenoising(labImg,None,10,7,21)
    return labImg/255.

def img_resize32(img):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = skimage.transform.resize(img,(32,32))
    return img

def img_equalize_adapthist(img):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = skimage.exposure.equalize_adapthist(img)
    return(img)    

test_images_fd = pd.read_csv("{}/sign_classes.csv".format(_test_images_folder))

#print(test_images_fd)
#print(len(test_images_fd))
#rint(test_images_fd.shape)

X_www = []
y_www = []


for i,r in test_images_fd.iterrows():
    if not len(y_www):
        y_www=np.array([r['class']])
    else:
        y_www=np.append(y_www, np.array([r['class']]))

    img = cv2.imread('{}/{}'.format(_test_images_folder,r['image_name']))
    print(r['image_name'],img.shape)
    img = img_resize32_cv2(img)
    img = img_rgb2LNorm(img)
    # img = img_resize32(img)
    img = img_equalize_adapthist(img)

    # cv2.imshow('image',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

#    break

    if not len(X_www):
        X_www=img[None,:]
    else:
        X_www=np.append(X_www, img[None,:], axis=0)

X_www = X_www[:,:,:,None]
print('X_www.shape:',X_www.shape)
print('y_www.shape:',y_www.shape)

with open('{}/n_ea_www.p'.format(_data_folder), 'wb') as handle:
    pickle.dump({'features':X_www, 'labels':y_www}, handle, protocol=pickle.HIGHEST_PROTOCOL)
