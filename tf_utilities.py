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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.python.platform

import numpy as np
import h5py

def zeroMeanUnitVariance(input_image):
	# zero mean unit variance
	augmented_image = np.zeros(input_image.shape)
	for ci in range(input_image.shape[0]):
		mn = np.mean(input_image[ci, ...])
		sd = np.std(input_image[ci, ...])
		augmented_image[ci, ...] = (input_image[ci, ...] - mn) / np.maximum(sd, 1e-5)
	return augmented_image

def compute_weights_multiClass(y,num_class):
	flat_y = y.reshape([-1, num_class])
	weight=np.zeros(num_class)
	for i in range(num_class):
		weight[i] = flat_y.sum() / flat_y[:,i].sum()
	return weight

def loadData_list_h5(X_train,y_train,num_channels):
	train_X = []
	train_y = []
	train_info = []
	for ii in range(len(X_train)):
		hf = h5py.File(str(X_train[ii]),'r')
		tmp_X = np.array(hf['data'])

		hf = h5py.File(str(y_train[ii]),'r')
		tmp_y = np.array(hf['data'])
		## here we add the background layer for softmax
		background = 1-(np.sum(tmp_y,axis=3)>0)
		addBG = np.zeros((tmp_y.shape[0],tmp_y.shape[1],tmp_y.shape[2],tmp_y.shape[3]+1))
		addBG[...,0]=background
		addBG[...,1:7] = tmp_y
		tmp_y=addBG

		if tmp_y.shape[0]!=384:
			print ("######")
		if tmp_y.shape[1]!=384:
			print ("######")
		print(ii,tmp_X.shape,tmp_y.shape,str(X_train[ii]),str(y_train[ii]))
		#tmp_y[tmp_y==3]=2
		train_X.append(tmp_X)
		train_y.append(tmp_y)
		train_info.append('Filename:%s'%(X_train[ii]))

	tmpp=np.asarray(train_X)
	#tmpp=tmpp[...,np.newaxis]
	return tmpp, np.asarray(train_y), train_info

def loadData_list_h5_single(X_train,y_train,num_channels):
	train_X = []
	train_y = []
	train_info = []

	hf = h5py.File(str(X_train),'r')
	tmp_X = np.array(hf['data'])

	hf = h5py.File(str(y_train),'r')
	tmp_y = np.array(hf['data'])
	## here we add the background layer for softmax
	background = 1-(np.sum(tmp_y,axis=3)>0)
	addBG = np.zeros((tmp_y.shape[0],tmp_y.shape[1],tmp_y.shape[2],tmp_y.shape[3]+1))
	addBG[...,0]=background
	addBG[...,1:7] = tmp_y
	tmp_y=addBG

	if tmp_y.shape[0]!=384:
		print ("######")
	if tmp_y.shape[1]!=384:
		print ("######")
	train_X.append(tmp_X)
	train_y.append(tmp_y)
	train_info.append('Filename:%s'%(X_train))

	tmpp=np.asarray(train_X)
	return tmpp, np.asarray(train_y), train_info

def loadData_list_h5_image(X_train,num_channels):
	train_X = []
	train_info = []
	for ii in range(len(X_train)):
		hf = h5py.File(str(X_train[ii]),'r')
		tmp_X = np.array(hf['data'])

		if tmp_X.shape[0]!=384:
			print ("######")
		if tmp_X.shape[1]!=384:
			print ("######")

		train_X.append(tmp_X)
		train_info.append('Filename:%s'%(X_train[ii]))

	tmpp=np.asarray(train_X)
	return tmpp, train_info

def crop_to_shape(data, shape):
    """
    Crops the array to the given image shape by removing the border (expects a tensor of shape [batches, nx, ny,nz, channels].

    :param data: the array to crop
    :param shape: the target shape
    """
    offset0 = (data.shape[1] - shape[1]) // 2
    offset1 = (data.shape[2] - shape[2]) // 2
    offset2 = (data.shape[3] - shape[3]) // 2

    if offset0 == 0 and offset1 == 0 and offset2 == 0:
        return data
    else:
    	return data[:, offset0:(-offset0), offset1:(-offset1),offset2:(-offset2),:]


