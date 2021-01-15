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
import sys

import numpy as np
import re
import time
import os
from functools import partial

import h5py
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from keras.utils import to_categorical
from pathlib import Path


tf.app.flags.DEFINE_boolean('restore', False, 'Whether to restore from previous model.')
tf.app.flags.DEFINE_float('lr', 0.00005, 'Initial learning rate.')
tf.app.flags.DEFINE_integer('feature', 16, 'Number of root features.')
tf.app.flags.DEFINE_string('model', '4atrous248', 'Model name.')
tf.app.flags.DEFINE_boolean('val', True, 'Whether to use validation.')
tf.app.flags.DEFINE_boolean('full_data', True, 'Whether to use full data set.')
tf.app.flags.DEFINE_float('dr', 1.0, 'Learning rate decay rate.')
tf.app.flags.DEFINE_integer('reso', 384, 'Image size.')
tf.app.flags.DEFINE_integer('slices', 160, 'Number Of Slices')
tf.app.flags.DEFINE_string('loss', 'wce', 'Loss name.')
tf.app.flags.DEFINE_integer('epoch', 400, 'Number of epochs.')
tf.app.flags.DEFINE_boolean('staircase', False, 'If True decay the learning rate at discrete intervals.')
tf.app.flags.DEFINE_integer('seed', 1234, 'Graph-level random seed.')
tf.app.flags.DEFINE_float('dropout', 1.0, 'Dropout rate when training.')
tf.app.flags.DEFINE_string('output_path', None, 'Name of output folder.')
tf.app.flags.DEFINE_boolean('resnet', False, 'Whether to use resnet shortcut.')
tf.app.flags.DEFINE_boolean('early_stopping', True, 'early stopping feature')
tf.app.flags.DEFINE_string('folder', './data', 'Data Folder')
tf.app.flags.DEFINE_integer('noImages', -1, 'how many images to train and validate')
tf.app.flags.DEFINE_float('switchAccuracy', 0.88, 'Training accuracy switch to Dice loss')
tf.app.flags.DEFINE_string('info', ' ', 'add some info to run')

FLAGS = tf.app.flags.FLAGS

switchAccuracy = FLAGS.switchAccuracy

num_classes = 7
num_channels = 1

def _get_cost(logits, batch_y, cost_name='dice', add_regularizers=None, class_weights=None):
    flat_logits = tf.reshape(logits, [-1, num_classes])
    flat_labels = tf.reshape(batch_y, [-1, num_classes])
    
    if cost_name == 'cross_entropy':
        if class_weights is not None:
            weight_map = tf.multiply(flat_labels, class_weights)
            weight_map = tf.reduce_sum(weight_map, axis=1)
            loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                            labels=flat_labels)
            weighted_loss = tf.multiply(loss_map, weight_map)
            loss = tf.reduce_mean(weighted_loss)
        else:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels))

    elif cost_name == 'dice':
        flat_logits = tf.nn.softmax(flat_logits)[:, 1]
        flat_labels = flat_labels[:, 1]

        inse = tf.reduce_sum(flat_logits*flat_labels)
        l = tf.reduce_sum(flat_logits*flat_logits)
        r = tf.reduce_sum(flat_labels*flat_labels)
        dice = 2 *(inse) / (l+r)
        loss = 1.0-tf.clip_by_value(dice,0,1-1e-10)

    elif cost_name == 'dice_multi':
        dice_multi = 0
        n_classes = num_classes
        for index in range(n_classes):
            flat_logits_ = tf.nn.softmax(flat_logits)[:, index]
            flat_labels_ = flat_labels[:, index]

            inse = tf.reduce_sum(flat_logits_*flat_labels_)
            l = tf.reduce_sum(flat_logits_*flat_logits_)
            r = tf.reduce_sum(flat_labels_*flat_labels_)
            dice = 2 *(inse) / (l+r)
            dice = tf.clip_by_value(dice,0,1-1e-10)

            dice_multi += dice

        loss = n_classes*1.0-dice_multi


    elif cost_name == 'dice_multi_noBG':
        dice_multi = 0
        n_classes = num_classes
        for index in range(1,n_classes):
            flat_logits_ = tf.nn.softmax(flat_logits)[:, index]
            flat_labels_ = flat_labels[:, index]

            inse = tf.reduce_sum(flat_logits_*flat_labels_)
            l = tf.reduce_sum(flat_logits_*flat_logits_)
            r = tf.reduce_sum(flat_labels_*flat_labels_)
            dice = 2 *(inse) / (l+r)
            dice = tf.clip_by_value(dice,0,1-1e-10)

            dice_multi += dice

        loss = (n_classes-1)*1.0-dice_multi

    return loss

def _get_acc(logits, batch_y, cost_name='dice', add_regularizers=None, class_weights=None):
    flat_logits = tf.reshape(logits, [-1, num_classes])
    flat_labels = tf.reshape(batch_y, [-1, num_classes])

    correct_prediction = tf.equal(tf.argmax(flat_logits,1), tf.argmax(flat_labels,1))
    correct_prediction = tf.boolean_mask(correct_prediction, tf.equal(flat_labels[:,0],0))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy

def _get_optimizer(start_learning_rate=0.0001, global_step=0, decay_steps=25, decay_rate=0.9):
    learning_rate = tf.train.exponential_decay(start_learning_rate,
                                               global_step,
                                               decay_steps,
                                               decay_rate,
                                               staircase=FLAGS.staircase)
    tf.summary.scalar('learning rate', learning_rate)
    optimizer=tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.995)
    return optimizer
    
def main(argv=None):
    # if no output path is given, create a new folder using flags
    res = 'res' if FLAGS.resnet else 'nores'
    if FLAGS.output_path is None:
        FLAGS.output_path = 'TrainedModels/' + '_'.join([time.strftime('%m%d_%H%M'),
                                    FLAGS.model,'wceSwitch%.2fDice_AccVal'%(switchAccuracy),
                                    res,
                                    FLAGS.loss, 
                                    'no' + str(FLAGS.noImages),
                                    'reso' + str(FLAGS.reso), 
                                    'features' + str(FLAGS.feature),
                                    'lr' + '{:.1e}'.format(FLAGS.lr), 
                                    'dr' + str(FLAGS.dropout)])

    if not os.path.exists(FLAGS.output_path):
        os.makedirs(FLAGS.output_path)
        
    # save flags into file
    with open(FLAGS.output_path + '/flags.txt', 'a') as f:
        f.write(str(FLAGS.flag_values_dict()))

    # set seeds for tensorflow and numpy
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    
    # placeholders
    batch_x = tf.placeholder(tf.float32, shape=(None, FLAGS.reso, FLAGS.reso, FLAGS.slices, 1), name='batch_x')
    batch_y = tf.placeholder(tf.float32, shape=(None, None, None, None, num_classes))
    
    keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
    global_step = tf.placeholder(tf.int32, shape=[])
    class_weights = tf.placeholder(tf.float32, shape=(num_classes))

    # choose the model
    inference_raw = {'4unet': models.inference_unet4, # the original architecture and use 4 layers
                     '4atrous248': partial(models.inference_atrous4, dilation_rates=[2,4,8])}[FLAGS.model]

    inference = partial(inference_raw, resnet=FLAGS.resnet)

    # get score and probability, add to summary
    score = inference(batch_x, features_root=FLAGS.feature, keep_prob=keep_prob, n_class=num_classes)
    logits = tf.nn.softmax(score)

    # get losses
    dice_cost = _get_cost(score, batch_y, cost_name='dice_multi')
    tf.summary.scalar('dice_loss', dice_cost)  
    dice_cost_noBG = _get_cost(score, batch_y, cost_name='dice_multi_noBG')
    tf.summary.scalar('dice_loss noBG', dice_cost_noBG)   

    cross_entropy = _get_cost(score, batch_y, cost_name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy) 

    weighted_cross_entropy = _get_cost(score, batch_y, cost_name='cross_entropy', class_weights=class_weights)
    tf.summary.scalar('weighted_cross_entropy',  weighted_cross_entropy)     

    if FLAGS.loss == 'wce': # weighted cross entropy
        cost = weighted_cross_entropy
    elif FLAGS.loss == 'dice': # dice
        cost = dice_cost
    elif FLAGS.loss == 'ce': # cross entropy
        cost = cross_entropy
    else:
        cost = dice_cost

    # get accuracy
    accuracy = _get_acc(score, batch_y)

    # set optimizer with learning rate and decay rate
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        with tf.name_scope('rms_optimizer'):
            optimizer = _get_optimizer(FLAGS.lr, global_step, decay_rate=FLAGS.dr)
            optimizer_dice = _get_optimizer(FLAGS.lr, global_step, decay_rate=FLAGS.dr)
        
            grads = optimizer.compute_gradients(cost)
            grads_dice = optimizer_dice.compute_gradients(dice_cost)

            train = optimizer.apply_gradients(grads)
            train_dice = optimizer_dice.apply_gradients(grads_dice)	  

    # get merged summaries
    merged = tf.summary.merge_all()

    # get losses & acc for training
    dice_cost_train = tf.placeholder(tf.float32, shape=[])
    dice_loss_train_summary = tf.summary.scalar('dice_loss_train', dice_cost_train)    

    cross_entropy_train = tf.placeholder(tf.float32, shape=[])
    cross_entropy_train_summary = tf.summary.scalar('cross_entropy_train', cross_entropy_train) 

    weighted_cross_entropy_train = tf.placeholder(tf.float32, shape=[])
    weighted_cross_entropy_train_summary = tf.summary.scalar('weighted_cross_entropy_train',  weighted_cross_entropy_train)    

    accuracy_train = tf.placeholder(tf.float32, shape=[])
    accuracy_train_summary = tf.summary.scalar('accuracy_train',  accuracy_train)  

    # get losses & acc for validation
    dice_cost_val = tf.placeholder(tf.float32, shape=[])
    dice_loss_val_summary = tf.summary.scalar('dice_loss_val', dice_cost_val)    

    cross_entropy_val = tf.placeholder(tf.float32, shape=[])
    cross_entropy_val_summary = tf.summary.scalar('cross_entropy_val', cross_entropy_val) 

    weighted_cross_entropy_val = tf.placeholder(tf.float32, shape=[])
    weighted_cross_entropy_val_summary = tf.summary.scalar('weighted_cross_entropy_val',  weighted_cross_entropy_val)   

    accuracy_val = tf.placeholder(tf.float32, shape=[])
    accuracy_val_summary = tf.summary.scalar('accuracy_val',  accuracy_val)  

    # load data
    #read multiple data
    dataFolder = FLAGS.folder + '/train'
    pathNifti = Path(dataFolder)

    X = []  # create an empty list
    for fileList in list(pathNifti.glob('**/*.im')):
        X.append(fileList)
    X = sorted(X)

    y = []  # create an empty list
    for fileList in list(pathNifti.glob('**/*.seg')):
        y.append(fileList)
    y = sorted(y)

    pathNifti = Path(FLAGS.folder + '/valid')

    X_v = []  # create an empty list
    for fileList in list(pathNifti.glob('**/*.im')):
        X_v.append(fileList)
    X_v = sorted(X_v)

    y_v = []  # create an empty list
    for fileList in list(pathNifti.glob('**/*.seg')):
        y_v.append(fileList)
    y_v = sorted(y_v)

    saver = tf.train.Saver(max_to_keep=0)

    # load mri data and segmentation maps for training
    if FLAGS.noImages ==-1:
        noOfFiles = len(X)
    else:
        noOfFiles = FLAGS.noImages
    list_X = list( X[i] for i in range(noOfFiles) )
    list_y = list( y[i] for i in range(noOfFiles) )

    X_train, y_train, train_info = tfut.loadData_list_h5(list_X,list_y,num_channels)
    print('Dataload is done')
    X_train = tfut.zeroMeanUnitVariance(X_train)
    weights_cross_entropy = tfut.compute_weights_multiClass(y_train,num_classes)
    del list_X, list_y

    # load mri data and segmentation maps for validation
    if FLAGS.noImages ==-1:
        noOfFiles = len(X_v)
    else:
        noOfFiles = FLAGS.noImages
    
    list_X = list( X_v[i] for i in range(noOfFiles) )
    list_y = list( y_v[i] for i in range(noOfFiles) )
    X_val, y_val, val_info = tfut.loadData_list_h5(list_X, list_y,num_channels)
    X_val = tfut.zeroMeanUnitVariance(X_val)
    del list_X, list_y

    X_train = X_train[...,np.newaxis]
    X_val = X_val[...,np.newaxis]

    # # resize data
    if FLAGS.reso != 384:
        input_size= X_train.shape[2]
        X_train = tfut.batch_resize(X_train, input_size=input_size, output_size=FLAGS.reso, order=3)
        y_train = tfut.batch_resize(y_train, input_size=input_size, output_size=FLAGS.reso, order=0)

        X_val = tfut.batch_resize(X_val, input_size=input_size, output_size=FLAGS.reso, order=3)
        y_val = tfut.batch_resize(y_val, input_size=input_size, output_size=FLAGS.reso, order=0)

    sample_size = X_train.shape[0]
    val_size = X_val.shape[0]

    # initialization for early stopping
    if FLAGS.early_stopping:
        best_acc = 0
        wait = 0
        patience = 500
        switchFlag = 1

    config = tf.ConfigProto()
    config.log_device_placement=False
    config.allow_soft_placement =True
    from tensorflow.python.client import device_lib

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        modelNo = 0
        if FLAGS.restore:
            ckpt = tf.train.get_checkpoint_state(FLAGS.output_path)
            model_path = ckpt.model_checkpoint_path
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.output_path))
            print('Model restored from file: %s' % model_path)
            tmp=re.findall('\d+', model_path)
            modelNo = int(tmp[-1])

        train_writer = tf.summary.FileWriter(FLAGS.output_path, sess.graph)

        start = time.clock()

        prediction = sess.run(score, feed_dict={batch_x: X_train[0:1], 
                                        batch_y: y_train[0:1],
                                        global_step:0,
                                        keep_prob:FLAGS.dropout,
                                        class_weights:weights_cross_entropy})
        pred_shape = prediction.shape

        offset0 = (y_train.shape[1] - pred_shape[1]) // 2
        offset1 = (y_train.shape[2] - pred_shape[2]) // 2
        offset2 = (y_train.shape[3] - pred_shape[3]) // 2

        if offset0 == 0 and offset1 == 0 and offset2 == 0:
            print('SAME padding')
        else:
            y_train = y_train[:, offset0:(-offset0), offset1:(-offset1),offset2:(-offset2),:]
            y_val = y_val[:, offset0:(-offset0), offset1:(-offset1),offset2:(-offset2),:]

        for epoch in range(modelNo+1, FLAGS.epoch+1):
            print('train epoch', epoch, 'sample_size', sample_size) 

            # shuffle data at the beginning of every epoch
            shuffled_idx = np.random.permutation(sample_size)
            wce_train, dice_train, ce_train, acc_train = [], [], [], []
            for j in range(sample_size):
                idx = shuffled_idx[j]
                i = (epoch - 1) * sample_size + j + 1

                # Whether to do left-right mirroring
                step = np.random.choice([1,-1]) 

                if switchFlag: 
                    _, loss, dice_loss, cross_entropy_loss, acc = sess.run([train, weighted_cross_entropy, dice_cost, cross_entropy, accuracy], 
                                                                        feed_dict={batch_x: X_train[idx:idx+1, :, :, ::step, :], 
                                                                                    batch_y: y_train[idx:idx+1, :, :, ::step, :],
                                                                                    global_step:epoch-1,
                                                                                    keep_prob:FLAGS.dropout,
                                                                                    class_weights:weights_cross_entropy})
                else:
                     _, loss, dice_loss, cross_entropy_loss, acc = sess.run([train_dice, weighted_cross_entropy, dice_cost, cross_entropy, accuracy], 
                                                                        feed_dict={batch_x: X_train[idx:idx+1, :, :, ::step, :], 
                                                                                    batch_y: y_train[idx:idx+1, :, :, ::step, :],
                                                                                    global_step:epoch-1,
                                                                                    keep_prob:FLAGS.dropout,
                                                                                    class_weights:weights_cross_entropy})

                wce_train.append(loss)
                dice_train.append(dice_loss)
                ce_train.append(cross_entropy_loss)
                acc_train.append(acc)

            # swithc to dice loss when the CE train accuracy is pretty good
            if np.mean(acc_train) > switchAccuracy:
                switchFlag = 0
                print('@@@@ switchtoDicein Epoch#:' ,epoch )

            print('training weighted loss:', np.mean(wce_train), \
                    ', cross entropy loss:', np.mean(ce_train), \
                    ', dice loss:', np.mean(dice_train), \
                    ', accuracy:', np.mean(acc_train))
            summary = sess.run(weighted_cross_entropy_train_summary, feed_dict={weighted_cross_entropy_train:np.mean(wce_train)})
            train_writer.add_summary(summary, epoch)
            summary = sess.run(dice_loss_train_summary, feed_dict={dice_cost_train:np.mean(dice_train)})
            train_writer.add_summary(summary, epoch)
            summary = sess.run(cross_entropy_train_summary, feed_dict={cross_entropy_train:np.mean(ce_train)})
            train_writer.add_summary(summary, epoch)
            summary = sess.run(accuracy_train_summary , feed_dict={accuracy_train:np.mean(acc_train)})
            train_writer.add_summary(summary, epoch)

            if FLAGS.val:
                summary = sess.run(merged, 
                                    feed_dict={batch_x: X_train[:1],
                                                batch_y: y_train[:1],
                                                global_step:epoch-1,
                                                keep_prob:1.0,
                                                class_weights:weights_cross_entropy})
                train_writer.add_summary(summary, epoch)

                wce_val, dice_val, ce_val, acc_val = [], [], [], []
                for j in range(val_size):
                    loss, dice_loss, cross_entropy_loss, acc = sess.run([weighted_cross_entropy, dice_cost, cross_entropy, accuracy], 
                                                                            feed_dict={batch_x: X_val[j:j+1], 
                                                                                        batch_y: y_val[j:j+1],
                                                                                        global_step:epoch-1,
                                                                                        keep_prob:1.0,
                                                                                        class_weights:weights_cross_entropy})
                    wce_val.append(loss)
                    dice_val.append(dice_loss)
                    ce_val.append(cross_entropy_loss)
                    acc_val.append(acc)

                summary = sess.run(weighted_cross_entropy_val_summary, feed_dict={weighted_cross_entropy_val:np.mean(wce_val)})
                train_writer.add_summary(summary, epoch)
                summary = sess.run(dice_loss_val_summary, feed_dict={dice_cost_val:np.mean(dice_val)})
                train_writer.add_summary(summary, epoch)
                summary = sess.run(cross_entropy_val_summary, feed_dict={cross_entropy_val:np.mean(ce_val)})
                train_writer.add_summary(summary, epoch)
                summary = sess.run(accuracy_val_summary, feed_dict={accuracy_val:np.mean(acc_val)})
                train_writer.add_summary(summary, epoch)  

                print('validation weighted loss:', np.mean(wce_val), \
                    ', cross entropy loss:', np.mean(ce_val), \
                    ', dice loss:', np.mean(dice_val), \
                    ', accuracy:', np.mean(acc_val))

                acc = np.mean(acc_val)
                if  acc - 1e-18 > best_acc:
                    best_acc, wait = acc, 0
                    saver.save(sess, FLAGS.output_path+'/model')
                    with open(FLAGS.output_path + '/SavedEpochNo.txt', 'w') as f:
                        f.write(str(epoch))
                else:
                    saver.save(sess, FLAGS.output_path+'/model_lastEpoch')
                    with open(FLAGS.output_path + '/SavedEpochNoLastEpoch.txt', 'w') as f:
                        f.write(str(epoch))
                    wait += 1
                    if wait > patience:
                        print("!!!!Early Stopping on EPOCH %d!!!!" % epoch)
                        break
                print("!!!!BEST: %f, wait %d !!!"%(best_acc, wait))


if __name__ == '__main__':
    tf.app.run()
