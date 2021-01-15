# ==============================================================================
# Copyright (C) 2020 Vladimir Juras, Ravinder Regatte and Cem M. Deniz
#
# This file is part of 2019_IWOAI_Challenge
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ==============================================================================
import tensorflow as tf
import tf_utilities as tfut
import tf_layers as tflay
import models
import nibabel as nib
import numpy as np
import re
import time
import os
from functools import partial
from pathlib import Path
import h5py
from scipy import ndimage
from scipy.spatial import distance

import glob
from sklearn.model_selection import StratifiedKFold
from sys import platform
from sklearn.preprocessing import label_binarize

tf.app.flags.DEFINE_string('model_path', "./InferenceModel", 'Name of output folder.')
tf.app.flags.DEFINE_string('data_folder', './data', 'Data Folder')
tf.app.flags.DEFINE_integer('cv', -1, 'which fold to run')
tf.app.flags.DEFINE_integer('feature', 16, 'which fold to run')
tf.app.flags.DEFINE_string('model', '4atrous248', 'Model name.')
tf.app.flags.DEFINE_integer('reso', 384, 'Image size.')
tf.app.flags.DEFINE_integer('slices', 160, 'Number Of Slices')
tf.app.flags.DEFINE_integer('seed', 1234, 'Graph-level random seed.')
tf.app.flags.DEFINE_boolean('resnet', False, 'Whether to use resnet shortcut.')
tf.app.flags.DEFINE_integer('noImages', -1, 'how many images to infer')
FLAGS = tf.app.flags.FLAGS

num_classes = 7
num_CV =1
num_channels = 1


def main(argv=None):

    print('OUT:: ',FLAGS.feature,FLAGS.seed, FLAGS.resnet,FLAGS.model,FLAGS.reso)

    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
            
    batch_x = tf.placeholder(tf.float32, shape=(None, FLAGS.reso, FLAGS.reso, FLAGS.slices, 1))
    batch_y = tf.placeholder(tf.float32, shape=(None, FLAGS.reso, FLAGS.reso, FLAGS.slices, num_classes))

    keep_prob = tf.placeholder(tf.float32, shape=[])
    class_weights = tf.placeholder(tf.float32, shape=(num_classes))

    # choose the model
    inference_raw = {'4unet': partial(models.inference_unet4),# the original architecture and use 4 layers 
                      # replace the convolution operations between down-convolution and up-convolution layers 
                      # by atrous convolution
                     '4atrous248': partial(models.inference_atrous4, n_class=num_classes, dilation_rates=[2,4,8])}[FLAGS.model]

    inference = partial(inference_raw, resnet=FLAGS.resnet)

    score = inference(batch_x, features_root=FLAGS.feature, keep_prob=keep_prob, n_class=num_classes)
    logits = tf.nn.softmax(score)

    # load dataset from folder
    dataFolder = FLAGS.data_folder + '/test'
    pathNifti = Path(dataFolder)

    X = []  # create an empty list
    for fileList in list(pathNifti.glob('**/*.im')):
        X.append(fileList)
    X = sorted(X)

    if FLAGS.noImages ==-1:
        noOfFiles = len(X)
    else:
        noOfFiles = FLAGS.noImages
    list_X = list( X[i] for i in range(noOfFiles) )
    n_samples = len(list_X)
    X_test, train_info = tfut.loadData_list_h5_image(list_X,num_channels)


    X_test = tfut.zeroMeanUnitVariance(X_test)
    X_test = X_test[...,np.newaxis]
    cv=1

    sample_size = X_test.shape[0]
    output_path = FLAGS.model_path
    
    # find the model to read
    fdr = Path(output_path)
    cpktFile = sorted(fdr.glob(('**/*.meta')))
    read_file = str((cpktFile[-1]))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=0)

        saver.restore(sess, read_file[:-5])
        print('Model restored from file: %s' % read_file[:-5])
        
        start = time.clock()
        y_out = np.zeros((FLAGS.reso, FLAGS.reso, FLAGS.slices, num_classes))
        for xi in range(sample_size):
            prob = sess.run(logits,
                                feed_dict={batch_x: X_test[xi:xi+1],
                                            keep_prob:1})
            y_out=prob

            winOut = np.zeros(y_out.shape)
            winOut[y_out[...,1]>0.5,...,1] =1
            winOut[y_out[...,2]>0.5,...,2] =2
            winOut[y_out[...,3]>0.5,...,3] =3
            winOut[y_out[...,4]>0.5,...,4] =4
            winOut[y_out[...,5]>0.5,...,5] =5
            winOut[y_out[...,6]>0.1,...,6] =6

            # place to keep only largest connected volume    
            if 1:
                for iii in range(1,7):
                    inn = winOut[...,iii]
                    all_labels, num_features = ndimage.label(inn)
                    print('Label #:',iii, num_features,'Number of Connected Volumes')
                    if num_features > 1:
                        volume = ndimage.sum(inn, all_labels, index=range(num_features+1))
                        print("Volume:", volume)
                        cem = all_labels == np.argmax(volume)
                        winOut[...,iii] = winOut[...,iii] * cem

            winOut =  np.sum(winOut,axis=4)
            winOut.astype(int)
            seg = np.pad(np.squeeze(winOut),((1,1),(1,1),(1,1)),'edge')

            #For the classes dimensions, the order for 4 classes are as the following: 
            #0 = femoral cartilage, 1 = tibial cartilage, 2 = patellar cartilage, and 3 = meniscus.
            #save as numpy array
            saveNumpy = np.zeros((384,384,160,4))
            saveNumpy[seg==1,...,0] = 1
            saveNumpy[seg==2,...,1] = 1
            saveNumpy[seg==3,...,1] = 1
            saveNumpy[seg==4,...,2] = 1
            saveNumpy[seg==5,...,3] = 1
            saveNumpy[seg==6,...,3] = 1
            savename= str(X[xi])
            fdr = Path('./InferenceResults/%s.npy' % (savename[-15:-3]))

            np.save(fdr, saveNumpy.astype(int), allow_pickle = False)
            

if __name__ == '__main__':
    tf.app.run()
