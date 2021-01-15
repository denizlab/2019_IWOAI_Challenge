
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
import tf_layers as tflay
from functools import partial


def inference_unet4(x, reg_c=0.1, keep_prob=0.5, channels=1, n_class=2, features_root=1, filter_size=3, pool_size=2, summaries=True, resnet=True):
    '''
    Creates a new convolutional net for the given parametrization.

    :param x: input tensor, shape [?,nx,ny,nz,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param n_class: number of output labels
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    '''

    print('Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{pool_size}'.format(layers=4,
                                                                                                           features=features_root,
                                                                                                           filter_size=filter_size,
                                                                                                           pool_size=pool_size))
    if not resnet:
        add_res = partial(tflay.add_res, skip=True)
    else:
        add_res = partial(tflay.add_res, skip=False)

    # Placeholder for the input image
    nx, ny, nz, channels = x.get_shape()[-4:]
    x_image = tf.reshape(x, tf.stack([-1,nx,ny,nz,channels]))
    shape_u0a = [nx, ny, nz]
    shape_u1a = [(n+1)//2 for n in shape_u0a]
    shape_u2a = [(n+1)//2 for n in shape_u1a]
    shape_u3a = [(n+1)//2 for n in shape_u2a]
    
    batch_size = tf.shape(x_image)[0]

    d0a = tflay.relu('relu_d0a', tflay.conv3d('conv_d0a', x_image, features_root, reg_constant=reg_c))
    d0b = tflay.relu('relu_d0b', add_res('res_d0b', tflay.conv3d('conv_d0b', d0a, features_root, reg_constant=reg_c), x_image, conv=False)) # 128 * 128 * 48, n

    d1a = tflay.max_pool('pool_d1a', d0b) # 64 * 64 * 24, n
    d1b = tflay.relu('relu_d1b', tflay.conv3d('conv_d1b', d1a, 2**1*features_root, reg_constant=reg_c)) # 64 * 64 * 24, 2n
    d1c = tflay.relu('relu_d1c', add_res('res_d1c', tflay.conv3d('conv_d1b-c', d1b, 2**1*features_root, reg_constant=reg_c), d1a)) # 64 * 64 * 24, 2n

    d2a = tflay.max_pool('pool_d2a', d1c) # 32 * 32 * 12, 2n
    d2b = tflay.relu('relu_d2b', tflay.conv3d('conv_d2b', d2a, 2**2*features_root, reg_constant=reg_c)) # 32 * 32 * 12, 4n
    d2c = tflay.relu('relu_d2c', add_res('res_d2c', tflay.conv3d('conv_d2b-c', d2b, 2**2*features_root, reg_constant=reg_c), d2a)) # 32 * 32 * 12, 4n

    d3a = tflay.max_pool('pool_d3a', d2c) # 16 * 16 * 6, 4n
    d3b = tflay.relu('relu_d3b', tflay.conv3d('conv_d3b', d3a, 2**3*features_root, reg_constant=reg_c)) # 16 * 16 * 6, 8n
    d3c = tflay.relu('relu_d3c', add_res('res_d3c', tflay.conv3d('conv_d3b-c', d3b, 2**3*features_root, reg_constant=reg_c), d3a)) # 16 * 16 * 6, 8n
    d3c = tflay.dropout('dropout_d3c', d3c, keep_prob)

    d4a = tflay.max_pool('pool_d4a', d3c) # 8 * 8 * 3, 8n
    d4b = tflay.relu('relu_d4b', tflay.conv3d('conv_d4b', d4a, 2**4*features_root, kernel_size=[3,3,1], reg_constant=reg_c)) # 8 * 8 * 3, 16n
    d4c = tflay.relu('relu_d4c', add_res('res_d4c', tflay.conv3d('conv_d4b-c', d4b, 2**4*features_root, reg_constant=reg_c), d4a)) # 8 * 8 * 3, 16n
    d4c = tflay.dropout('dropout_d4c', d4c, keep_prob)

    u3a = tflay.concat('concat_u3a', tflay.relu('relu_u3a', tflay.upconv3d('upconv_u3a', d4c, 2**3*features_root, shape_u3a, reg_constant=reg_c)), d3c) # 16 * 16 * 6, 16n
    u3b = tflay.relu('relu_u3b', tflay.conv3d('conv_u3a-b', u3a, 2**3*features_root, reg_constant=reg_c)) # 16 * 16 * 6, 8n
    u3c = tflay.relu('relu_u3c', add_res('res_u3c', tflay.conv3d('conv_u3b-c', u3b, 2**3*features_root, reg_constant=reg_c), u3a)) # 16 * 16 * 6, 8n

    u2a = tflay.concat('concat_u2a', tflay.relu('relu_u2a', tflay.upconv3d('upconv_u2a', u3c, 2**2*features_root, shape_u2a, reg_constant=reg_c)), d2c) # 32 * 32 * 12, 8n
    u2b = tflay.relu('relu_u2b', tflay.conv3d('conv_u2a-b', u2a, 2**2*features_root, reg_constant=reg_c)) # 32 * 32 * 12, 4n
    u2c = tflay.relu('relu_u2c', add_res('res_u2c', tflay.conv3d('conv_u2b-c', u2b, 2**2*features_root, reg_constant=reg_c), u2a)) # 32 * 32 * 12, 4n

    u1a = tflay.concat('concat_u1a', tflay.relu('relu_u1a', tflay.upconv3d('upconv_u1a', u2c, 2**1*features_root, shape_u1a, reg_constant=reg_c)), d1c) # 64 * 64 * 24, 4n
    u1b = tflay.relu('relu_u1b', tflay.conv3d('conv_u1a-b', u1a, 2**1*features_root, reg_constant=reg_c)) # 64 * 64 * 24, 2n
    u1c = tflay.relu('relu_u1c', add_res('res_u1c', tflay.conv3d('conv_u1b-c', u1b, 2**1*features_root, reg_constant=reg_c), u1a)) # 64 * 64 * 24, 2n

    u0a = tflay.concat('concat_u0a', tflay.relu('relu_u0a', tflay.upconv3d('upconv_u0a', u1c, 2**0*features_root, shape_u0a, reg_constant=reg_c)), d0b) # 128 * 128 * 48, 2n
    u0b = tflay.relu('relu_u0b', tflay.conv3d('conv_u0a-b', u0a, 2**0*features_root, reg_constant=reg_c)) # 128 * 128 * 48, n
    u0c = tflay.relu('relu_u0c', add_res('res_u0c', tflay.conv3d('conv_u0b-c', u0b, 2**0*features_root, reg_constant=reg_c, padding='VALID'), u0a)) # 128 * 128 * 48, n

    score = tflay.relu('relu_result', tflay.conv3d('conv_result', u0c, n_class, kernel_size=[1,1,1], reg_constant=reg_c))

    return score

def inference_atrous4(x, reg_c=0.1, keep_prob=0.5, channels=1, n_class=2, features_root=1, filter_size=3, pool_size=2, dilation_rates=[2], summaries=True, resnet=True):
    '''
    Creates a new convolutional net for the given parametrization.

    :param x: input tensor, shape [?,nx,ny,nz,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param n_class: number of output labels
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    '''

    print('Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{pool_size}'.format(layers=5,
                                                                                                           features=features_root,
                                                                                                           filter_size=filter_size,
                                                                                                           pool_size=pool_size))
    if not resnet:
        add_res = partial(tflay.add_res, skip=True)
    else:
        add_res = partial(tflay.add_res, skip=False)

    # Placeholder for the input image
    nx, ny, nz, channels = x.get_shape()[-4:]
    x_image = tf.reshape(x, tf.stack([-1,nx,ny,nz,channels]))
    shape_u0a = [nx, ny, nz]
    shape_u1a = [(n+1)//2 for n in shape_u0a]
    shape_u2a = [(n+1)//2 for n in shape_u1a]
    shape_u3a = [(n+1)//2 for n in shape_u2a]
    shape_u4a = [(n+1)//2 for n in shape_u3a]

    batch_size = tf.shape(x_image)[0]

    d0a = tflay.relu('relu_d0a', tflay.conv3d('conv_d0a', x_image, features_root, reg_constant=reg_c))
    d0b = tflay.relu('relu_d0b', add_res('res_d0b', tflay.conv3d('conv_d0b', d0a, features_root, reg_constant=reg_c), x_image, conv=False)) # 128 * 128 * 48, n

    d1a = tflay.max_pool('pool_d1a', d0b) # 64 * 64 * 24, n
    d1b = tflay.relu('relu_d1b', tflay.conv3d('conv_d1b', d1a, 2**1*features_root, reg_constant=reg_c)) # 64 * 64 * 24, 2n
    d1c = tflay.relu('relu_d1c', add_res('res_d1c', tflay.conv3d('conv_d1b-c', d1b, 2**1*features_root, reg_constant=reg_c), d1a)) # 64 * 64 * 24, 2n

    d2a = tflay.max_pool('pool_d2a', d1c) # 32 * 32 * 12, 2n
    d2b = tflay.relu('relu_d2b', tflay.conv3d('conv_d2b', d2a, 2**2*features_root, reg_constant=reg_c)) # 32 * 32 * 12, 4n
    d2c = tflay.relu('relu_d2c', add_res('res_d2c', tflay.conv3d('conv_d2b-c', d2b, 2**2*features_root, reg_constant=reg_c), d2a)) # 32 * 32 * 12, 4n

    d3a = tflay.max_pool('pool_d3a', d2c) # 16 * 16 * 6, 4n
    d3b = tflay.relu('relu_d3b', tflay.conv3d('conv_d3b', d3a, 2**3*features_root, reg_constant=reg_c)) # 16 * 16 * 6, 8n
    d3c = tflay.relu('relu_d3c', add_res('res_d3c', tflay.conv3d('conv_d3b-c', d3b, 2**3*features_root, reg_constant=reg_c), d3a)) # 16 * 16 * 6, 8n

    d4a = tflay.max_pool('pool_d4a', d3c) # 8 * 8 * 3, 8n
    d4b = tflay.relu('relu_d4b', tflay.conv3d('conv_d4b', d4a, 2**4*features_root, reg_constant=reg_c)) # 8 * 8 * 3, 16n
    d4c = tflay.relu('relu_d4c', add_res('res_d4c', tflay.conv3d('conv_d4b-c', d4b, 2**4*features_root, reg_constant=reg_c), d4a)) # 8 * 8 * 3, 16n
    d4c = tflay.dropout('dropout_d4c', d4c, keep_prob)

    bs = [d4c]
    for i, dilation_rate in enumerate(dilation_rates):
        name = 'b' + str(i) + 'a'
        tmp = tflay.relu('relu_'+name, tflay.atrousconv3d('atrsconv_'+name, d4c, 2**4*features_root, dilation_rate=[dilation_rate,dilation_rate,1], reg_constant=reg_c)) # 8 * 8 * 3, 16n
        bs.append(tmp)
    bna = tflay.multiconcat('concat_bna', bs)
    bna = tflay.dropout('dropout_bna', bna, keep_prob)

    u3a = tflay.concat('concat_u3a', tflay.relu('relu_u3a', tflay.upconv3d('upconv_u3a', bna, 2**3*features_root, shape_u3a, reg_constant=reg_c)), d3c) # 16 * 16 * 6, 16n
    u3b = tflay.relu('relu_u3b', tflay.conv3d('conv_u3a-b', u3a, 2**3*features_root, reg_constant=reg_c)) # 16 * 16 * 6, 8n
    u3c = tflay.relu('relu_u3c', add_res('res_u3c', tflay.conv3d('conv_u3b-c', u3b, 2**3*features_root, reg_constant=reg_c), u3a)) # 16 * 16 * 6, 8n

    u2a = tflay.concat('concat_u2a', tflay.relu('relu_u2a', tflay.upconv3d('upconv_u2a', u3c, 2**2*features_root, shape_u2a, reg_constant=reg_c)), d2c) # 32 * 32 * 12, 8n
    u2b = tflay.relu('relu_u2b', tflay.conv3d('conv_u2a-b', u2a, 2**2*features_root, reg_constant=reg_c)) # 32 * 32 * 12, 4n
    u2c = tflay.relu('relu_u2c', add_res('res_u2c', tflay.conv3d('conv_u2b-c', u2b, 2**2*features_root, reg_constant=reg_c), u2a)) # 32 * 32 * 12, 4n

    u1a = tflay.concat('concat_u1a', tflay.relu('relu_u1a', tflay.upconv3d('upconv_u1a', u2c, 2**1*features_root, shape_u1a, reg_constant=reg_c)), d1c) # 64 * 64 * 24, 4n
    u1b = tflay.relu('relu_u1b', tflay.conv3d('conv_u1a-b', u1a, 2**1*features_root, reg_constant=reg_c)) # 64 * 64 * 24, 2n
    u1c = tflay.relu('relu_u1c', add_res('res_u1c', tflay.conv3d('conv_u1b-c', u1b, 2**1*features_root, reg_constant=reg_c), u1a)) # 64 * 64 * 24, 2n

    u0a = tflay.concat('concat_u0a', tflay.relu('relu_u0a', tflay.upconv3d('upconv_u0a', u1c, 2**0*features_root, shape_u0a, reg_constant=reg_c)), d0b) # 128 * 128 * 48, 2n
    u0b = tflay.relu('relu_u0b', tflay.conv3d('conv_u0a-b', u0a, 2**0*features_root, reg_constant=reg_c)) # 128 * 128 * 48, n
    u0c = tflay.relu('relu_u0c', add_res('res_u0c', tflay.conv3d('conv_u0b-c', u0b, 2**0*features_root, reg_constant=reg_c, padding='VALID'), u0a)) # 128 * 128 * 48, n

    score = tflay.relu('relu_result', tflay.conv3d('conv_result', u0c, n_class, kernel_size=[1,1,1], reg_constant=reg_c))

    return score
