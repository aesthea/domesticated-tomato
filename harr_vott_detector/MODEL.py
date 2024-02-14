# Copyright 2019 The TensorFlow Authors, Pavel Yakubovskiy, Björn Barz. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for EfficientNet model.

[1] Mingxing Tan, Quoc V. Le
  EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
  ICML'19, https://arxiv.org/abs/1905.11946
"""

# Code of this model implementation is mostly written by
# Björn Barz ([@Callidior](https://github.com/Callidior))

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import string
import collections

import tensorflow
#import tensorflow.keras.layers as layers
from six.moves import xrange
from tensorflow.keras import layers 


BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])
# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

DEFAULT_BLOCKS_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
              expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25)
]

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}



def round_filters(filters, width_coefficient, depth_divisor):
    """Round number of filters based on width multiplier."""

    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""

    return int(math.ceil(depth_coefficient * repeats))


def mb_conv_block(inputs, block_args, activation, drop_rate=None, prefix='', ):
    """Mobile Inverted Residual Bottleneck."""

    has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)
    bn_axis = 3

    # workaround over non working dropout with None in noise_shape in tf.keras
    Dropout = layers.Dropout

    # Expansion phase
    filters = block_args.input_filters * block_args.expand_ratio
    if block_args.expand_ratio != 1:
        x = layers.Conv2D(filters, 1,
                          padding='same',
                          use_bias=False,
                          kernel_initializer=CONV_KERNEL_INITIALIZER,
                          name=prefix + 'expand_conv')(inputs)
        x = layers.BatchNormalization(axis=bn_axis, name=prefix + 'expand_bn')(x)
        x = layers.Activation(activation, name=prefix + 'expand_activation')(x)
    else:
        x = inputs

    # Depthwise Convolution
    x = layers.DepthwiseConv2D(block_args.kernel_size,
                               strides=block_args.strides,
                               padding='same',
                               use_bias=False,
                               depthwise_initializer=CONV_KERNEL_INITIALIZER,
                               name=prefix + 'dwconv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=prefix + 'bn')(x)
    x = layers.Activation(activation, name=prefix + 'activation')(x)

    # Squeeze and Excitation phase
    if has_se:
        num_reduced_filters = max(1, int(
            block_args.input_filters * block_args.se_ratio
        ))
        se_tensor = layers.GlobalAveragePooling2D(name=prefix + 'se_squeeze')(x)

        target_shape = (1, 1, filters) 
        se_tensor = layers.Reshape(target_shape, name=prefix + 'se_reshape')(se_tensor)
        se_tensor = layers.Conv2D(num_reduced_filters, 1,
                                  activation=activation,
                                  padding='same',
                                  use_bias=True,
                                  kernel_initializer=CONV_KERNEL_INITIALIZER,
                                  name=prefix + 'se_reduce')(se_tensor)
        se_tensor = layers.Conv2D(filters, 1,
                                  activation='sigmoid',
                                  padding='same',
                                  use_bias=True,
                                  kernel_initializer=CONV_KERNEL_INITIALIZER,
                                  name=prefix + 'se_expand')(se_tensor)

        x = layers.multiply([x, se_tensor], name=prefix + 'se_excite')

    # Output phase
    x = layers.Conv2D(block_args.output_filters, 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name=prefix + 'project_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=prefix + 'project_bn')(x)
    if block_args.id_skip and all(
            s == 1 for s in block_args.strides
    ) and block_args.input_filters == block_args.output_filters:
        if drop_rate and (drop_rate > 0):
            x = Dropout(drop_rate,
                        noise_shape=(None, 1, 1, 1),
                        name=prefix + 'drop')(x)
        x = layers.add([x, inputs], name=prefix + 'add')

    return x


def EfficientNet(width_coefficient,
                 depth_coefficient,
                 default_resolution,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 blocks_args=DEFAULT_BLOCKS_ARGS,
                 model_name='efficientnet',
                 include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 classes=1000,
                 **kwargs):
    """Instantiates the EfficientNet architecture using given scaling coefficients.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        width_coefficient: float, scaling coefficient for network width.
        depth_coefficient: float, scaling coefficient for network depth.
        default_resolution: int, default input image size.
        dropout_rate: float, dropout rate before final classifier layer.
        drop_connect_rate: float, dropout rate at skip connections.
        depth_divisor: int.
        blocks_args: A list of BlockArgs to construct block modules.
        model_name: string, model name.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False.
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    bn_axis = 3
    activation = "relu"
    
    # Build stem

    img_input = tensorflow.keras.Input(tensor=input_tensor, shape=input_shape)
    x = img_input
    x = layers.Conv2D(round_filters(32, width_coefficient, depth_divisor), 3,
                      strides=(2, 2),
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='stem_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='stem_bn')(x)
    x = layers.Activation(activation, name='stem_activation')(x)

    # Build blocks
    num_blocks_total = sum(round_repeats(block_args.num_repeat,
                                         depth_coefficient) for block_args in blocks_args)
    block_num = 0
    for idx, block_args in enumerate(blocks_args):
        assert block_args.num_repeat > 0
        # Update block input and output filters based on depth multiplier.
        block_args = block_args._replace(
            input_filters=round_filters(block_args.input_filters,
                                        width_coefficient, depth_divisor),
            output_filters=round_filters(block_args.output_filters,
                                         width_coefficient, depth_divisor),
            num_repeat=round_repeats(block_args.num_repeat, depth_coefficient))

        # The first block needs to take care of stride and filter size increase.
        drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
        x = mb_conv_block(x, block_args,
                          activation=activation,
                          drop_rate=drop_rate,
                          prefix='block{}a_'.format(idx + 1))
        block_num += 1
        if block_args.num_repeat > 1:
            # pylint: disable=protected-access
            block_args = block_args._replace(
                input_filters=block_args.output_filters, strides=[1, 1])
            # pylint: enable=protected-access
            for bidx in xrange(block_args.num_repeat - 1):
                drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
                block_prefix = 'block{}{}_'.format(
                    idx + 1,
                    string.ascii_lowercase[bidx + 1]
                )
                x = mb_conv_block(x, block_args,
                                  activation=activation,
                                  drop_rate=drop_rate,
                                  prefix=block_prefix)
                block_num += 1

    # Build top
    x = layers.Conv2D(round_filters(1280, width_coefficient, depth_divisor), 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='top_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='top_bn')(x)
    x = layers.Activation(activation, name='top_activation')(x)
    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        if dropout_rate and dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name='top_dropout')(x)
        x = layers.Dense(classes,
                         activation='softmax',
                         kernel_initializer=DENSE_KERNEL_INITIALIZER,
                         name='probs')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = tensorflow.keras.utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = tensorflow.keras.Model(inputs, x, name=model_name)

    return model

def EfficientNetS1(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        **kwargs
):
    return EfficientNet(
        0.5, 1.0, 128, 0.2,
        model_name='efficientnet-S1',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )

def EfficientNetS0(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        **kwargs
):
    return EfficientNet(
        0.7, 1.0, 128, 0.2,
        model_name='efficientnet-S0',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )

def EfficientNetB0(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        **kwargs
):
    return EfficientNet(
        1.0, 1.0, 224, 0.2,
        model_name='efficientnet-b0',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )


def EfficientNetB1(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        **kwargs
):
    return EfficientNet(
        1.0, 1.1, 240, 0.2,
        model_name='efficientnet-b1',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )


def EfficientNetB2(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(
        1.1, 1.2, 260, 0.3,
        model_name='efficientnet-b2',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )


def EfficientNetB3(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(
        1.2, 1.4, 300, 0.3,
        model_name='efficientnet-b3',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )


def EfficientNetB4(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        **kwargs
):
    return EfficientNet(
        1.4, 1.8, 380, 0.4,
        model_name='efficientnet-b4',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )


def EfficientNetB5(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        **kwargs
):
    return EfficientNet(
        1.6, 2.2, 456, 0.4,
        model_name='efficientnet-b5',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )


def EfficientNetB6(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        **kwargs
):
    return EfficientNet(
        1.8, 2.6, 528, 0.5,
        model_name='efficientnet-b6',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )


def EfficientNetB7(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        **kwargs
):
    return EfficientNet(
        2.0, 3.1, 600, 0.5,
        model_name='efficientnet-b7',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )


def EfficientNetL2(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        **kwargs
):
    return EfficientNet(
        4.3, 5.3, 800, 0.5,
        model_name='efficientnet-l2',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )


setattr(EfficientNetB0, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB1, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB2, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB3, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB4, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB5, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB6, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB7, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetL2, '__doc__', EfficientNet.__doc__)


tf = tensorflow
def conv_block(x, filters, strides, KERNELS, activation = 'relu'):
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=KERNELS, strides=strides, padding='same', activation = activation)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    return x

def edet(input_size = 256, num_channels = 1, num_classes = 100, items = 3, dropout = 0.0, bi = 0, backbone = "B1"):
    print("input", input_size, "channel", num_channels, "classtags", num_classes, "region", items, "fpn mode", bi, "backbone", backbone)
    x_in = tf.keras.Input(shape=[input_size, input_size, num_channels])
    if backbone == "B0":
        backbone = tf.keras.applications.efficientnet.EfficientNetB0(input_tensor = x_in, include_top = False, weights = None,input_shape = (input_size, input_size, num_channels), classes = num_classes)
        p0 = backbone.layers[0].output #(None, 256, 256, 1)
        p1 = backbone.layers[16].output #(None, 128, 128, 16)
        p2 = backbone.layers[45].output #(None, 64, 64, 24)
        p3 = backbone.layers[74].output #(None, 32, 32, 40)
        p4 = backbone.layers[118].output #(None, 16, 16, 80)
        p5 = backbone.layers[161].output #(None, 16, 16, 112)
        p6 = backbone.layers[220].output #(None, 8, 8, 192)
        p7 = backbone.layers[233].output #(None, 8, 8, 320)
    elif backbone == "B1":
        backbone = tf.keras.applications.efficientnet.EfficientNetB1(input_tensor = x_in, include_top = False, weights = None,input_shape = (input_size, input_size, num_channels), classes = num_classes)
        p0 = backbone.layers[0].output 
        p1 = backbone.layers[28].output
        p2 = backbone.layers[72].output 
        p3 = backbone.layers[116].output
        p4 = backbone.layers[175].output
        p5 = backbone.layers[233].output
        p6 = backbone.layers[277].output
        p7 = backbone.layers[335].output
    elif backbone == "B2":
        backbone = tf.keras.applications.efficientnet.EfficientNetB2(input_tensor = x_in, include_top = False, weights = None,input_shape = (input_size, input_size, num_channels), classes = num_classes)
        p0 = backbone.layers[0].output #(0, 'input_1', TensorShape([None, 256, 256, 3]))
        p1 = backbone.layers[28].output #(28, 'block1b_add', TensorShape([None, 128, 128, 16]))
        p2 = backbone.layers[72].output #(72, 'block2c_add', TensorShape([None, 64, 64, 24]))
        p3 = backbone.layers[116].output #(116, 'block3c_add', TensorShape([None, 32, 32, 48]))
        p4 = backbone.layers[175].output #(175, 'block4d_add', TensorShape([None, 16, 16, 88]))
        p5 = backbone.layers[233].output #(233, 'block5d_add', TensorShape([None, 16, 16, 120]))
        p6 = backbone.layers[307].output #(307, 'block6e_add', TensorShape([None, 8, 8, 208]))
        p7 = backbone.layers[335].output #(335, 'block7b_add', TensorShape([None, 8, 8, 352]))
    elif backbone == "B3":
        backbone = tf.keras.applications.efficientnet.EfficientNetB3(input_tensor = x_in, include_top = False, weights = None,input_shape = (input_size, input_size, num_channels), classes = num_classes)
        p0 = backbone.layers[0].output #(0, 'input_1', TensorShape([None, 256, 256, 3]))
        p1 = backbone.layers[28].output #(28, 'block1b_add', TensorShape([None, 128, 128, 24]))
        p2 = backbone.layers[72].output #(72, 'block2c_add', TensorShape([None, 64, 64, 32]))
        p3 = backbone.layers[116].output #(116, 'block3c_add', TensorShape([None, 32, 32, 48]))
        p4 = backbone.layers[190].output #(190, 'block4e_add', TensorShape([None, 16, 16, 96]))
        p5 = backbone.layers[263].output #(263, 'block5e_add', TensorShape([None, 16, 16, 136]))
        p6 = backbone.layers[352].output #(352, 'block6f_add', TensorShape([None, 8, 8, 232]))
        p7 = backbone.layers[380].output #(380, 'block7b_add', TensorShape([None, 8, 8, 384]))
    elif backbone == "B4":
        backbone = tf.keras.applications.efficientnet.EfficientNetB4(input_tensor = x_in, include_top = False, weights = None,input_shape = (input_size, input_size, num_channels), classes = num_classes)
        p0 = backbone.layers[0].output #(0, 'input_1', TensorShape([None, 256, 256, 3]))
        p1 = backbone.layers[28].output #(28, 'block1b_add', TensorShape([None, 128, 128, 24]))
        p2 = backbone.layers[87].output #(87, 'block2d_add', TensorShape([None, 64, 64, 32]))
        p3 = backbone.layers[146].output #(146, 'block3d_add', TensorShape([None, 32, 32, 56]))
        p4 = backbone.layers[235].output #(235, 'block4f_add', TensorShape([None, 16, 16, 112]))
        p5 = backbone.layers[323].output #(323, 'block5f_add', TensorShape([None, 16, 16, 160]))
        p6 = backbone.layers[442].output #(442, 'block6h_add', TensorShape([None, 8, 8, 272]))
        p7 = backbone.layers[470].output #(470, 'block7b_add', TensorShape([None, 8, 8, 448]))
    elif backbone == "B5":
        backbone = tf.keras.applications.efficientnet.EfficientNetB5(input_tensor = x_in, include_top = False, weights = None,input_shape = (input_size, input_size, num_channels), classes = num_classes)
        p0 = backbone.layers[0].output #(0, 'input_1', TensorShape([None, 256, 256, 3]))
        p1 = backbone.layers[40].output #(40, 'block1c_add', TensorShape([None, 128, 128, 24]))
        p2 = backbone.layers[114].output #(114, 'block2e_add', TensorShape([None, 64, 64, 40]))
        p3 = backbone.layers[188].output #(188, 'block3e_add', TensorShape([None, 32, 32, 64]))
        p4 = backbone.layers[292].output #(292, 'block4g_add', TensorShape([None, 16, 16, 128]))
        p5 = backbone.layers[395].output #(395, 'block5g_add', TensorShape([None, 16, 16, 176]))
        p6 = backbone.layers[529].output #(529, 'block6i_add', TensorShape([None, 8, 8, 304]))
        p7 = backbone.layers[572].output #(572, 'block7c_add', TensorShape([None, 8, 8, 512]))
    elif backbone == "B6":
        backbone = tf.keras.applications.efficientnet.EfficientNetB6(input_tensor = x_in, include_top = False, weights = None,input_shape = (input_size, input_size, num_channels), classes = num_classes)
        p0 = backbone.layers[0].output #(0, 'input_1', TensorShape([None, 256, 256, 3]))
        p1 = backbone.layers[40].output #(40, 'block1c_add', TensorShape([None, 128, 128, 32]))
        p2 = backbone.layers[129].output #(129, 'block2f_add', TensorShape([None, 64, 64, 40]))
        p3 = backbone.layers[218].output #(218, 'block3f_add', TensorShape([None, 32, 32, 72]))
        p4 = backbone.layers[337].output #(337, 'block4h_add', TensorShape([None, 16, 16, 144]))
        p5 = backbone.layers[455].output #(455, 'block5h_add', TensorShape([None, 16, 16, 200]))
        p6 = backbone.layers[619].output #(619, 'block6k_add', TensorShape([None, 8, 8, 344]))
        p7 = backbone.layers[662].output #(662, 'block7c_add', TensorShape([None, 8, 8, 576]))
    elif backbone == "B7":
        backbone = tf.keras.applications.efficientnet.EfficientNetB7(input_tensor = x_in, include_top = False, weights = None,input_shape = (input_size, input_size, num_channels), classes = num_classes)
        p0 = backbone.layers[0].output #(0, 'input_1', TensorShape([None, 256, 256, 3]))
        p1 = backbone.layers[52].output #(52, 'block1d_add', TensorShape([None, 128, 128, 32]))
        p2 = backbone.layers[156].output #(156, 'block2g_add', TensorShape([None, 64, 64, 48]))
        p3 = backbone.layers[260].output #(260, 'block3g_add', TensorShape([None, 32, 32, 80]))
        p4 = backbone.layers[409].output #(409, 'block4j_add', TensorShape([None, 16, 16, 160]))
        p5 = backbone.layers[557].output #(557, 'block5j_add', TensorShape([None, 16, 16, 224]))
        p6 = backbone.layers[751].output #(751, 'block6m_add', TensorShape([None, 8, 8, 384]))
        p7 = backbone.layers[809].output #(809, 'block7d_add', TensorShape([None, 8, 8, 640]))
    elif backbone == "S0":
        backbone = EfficientNetS0(input_tensor = x_in, include_top = False, weights = None,input_shape = (input_size, input_size, num_channels), classes = num_classes)
        p0 = backbone.layers[0].output #(None, 256, 256, 1)
        p1 = backbone.layers[12].output #12 block1a_project_conv (None, 128, 128, 16)
        p2 = backbone.layers[41].output #41 block2b_add (None, 64, 64, 16)
        p3 = backbone.layers[69].output #69 block3b_add (None, 32, 32, 32)
        p4 = backbone.layers[112].output #112 block4c_add (None, 16, 16, 56)
        p5 = backbone.layers[155].output #155 block5c_add (None, 16, 16, 80)
        p6 = backbone.layers[213].output #213 block6d_add (None, 8, 8, 136)
        p7 = backbone.layers[226].output #226 block7a_project_bn (None, 8, 8, 224)

    elif backbone == "S1":
        backbone = EfficientNetS1(input_tensor = x_in, include_top = False, weights = None,input_shape = (input_size, input_size, num_channels), classes = num_classes)
        p0 = backbone.layers[0].output #(None, 256, 256, 1)
        p1 = backbone.layers[12].output #12 block1a_project_conv (None, 128, 128, 16)
        p2 = backbone.layers[41].output #41 block2b_add (None, 64, 64, 16)
        p3 = backbone.layers[69].output #69 block3b_add (None, 32, 32, 24)
        p4 = backbone.layers[112].output #112 block4c_add (None, 16, 16, 40)
        p5 = backbone.layers[155].output #155 block5c_add (None, 16, 16, 56)
        p6 = backbone.layers[213].output #213 block6d_add (None, 8, 8, 96)
        p7 = backbone.layers[226].output #226 block7a_project_bn (None, 8, 8, 160)


    #print(p1.shape, p2.shape, p3.shape, p5.shape, p7.shape)
    o1, o2, o3, o4, o5 = FPN(p1, p2, p3, p5, p7, 3, 'relu', dropout, bi)
    #print(o1.shape, o2.shape, o3.shape, o4.shape, o5.shape)
    o1, o2, o3, o4, o5 = FPN(o1, o2, o3, o4, o5, 3, 'relu', dropout, bi)
    #print(o1.shape, o2.shape, o3.shape, o4.shape, o5.shape)
    c1, c2, c3, c4, c5 = FPN(o1, o2, o3, o4, o5, 3, 'relu', dropout, bi)
    L1 = tf.keras.layers.GlobalAveragePooling2D()(c1)
    L2 = tf.keras.layers.GlobalAveragePooling2D()(c2)
    L3 = tf.keras.layers.GlobalAveragePooling2D()(c3)
    L4 = tf.keras.layers.GlobalAveragePooling2D()(c4)
    L5 = tf.keras.layers.GlobalAveragePooling2D()(c5)

    r = tf.keras.layers.Concatenate(axis=-1)([L1, L2, L3, L4, L5])
    r = tf.keras.layers.Dropout(dropout)(r)
    b = tf.keras.layers.Dense(4 * items)(r)
    b = tf.keras.layers.Reshape([items, 4])(b)
    b = tf.keras.layers.Activation('sigmoid')(b)
    regression = tf.keras.layers.Layer(name = "regression")(b)

    c = tf.keras.layers.Dense(num_classes * items)(r)
    c = tf.keras.layers.Reshape([items, num_classes])(c)
    c = tf.keras.layers.BatchNormalization(axis=-1)(c)
    c = tf.keras.layers.Activation('softmax')(c)
    classification = tf.keras.layers.Layer(name = "classification")(c)
    
    model = tf.keras.Model(x_in, [classification, regression])
    return model


def FPN(i1, i2, i3, i4, i5, KERNELS = 3, end_activation = "relu", dropout = 0.2, bi = False):
    pool_size_2_1 = i1.shape[1] // i2.shape[1]
    pool_size_3_2 = i2.shape[1] // i3.shape[1]
    pool_size_4_3 = i3.shape[1] // i4.shape[1]
    pool_size_5_4 = i4.shape[1] // i5.shape[1]
    u2_1 = tf.keras.layers.Conv2DTranspose(i1.shape[-1], KERNELS, strides=(pool_size_2_1, pool_size_2_1), padding="same", activation = "relu")(i2)
    u3_2 = tf.keras.layers.Conv2DTranspose(i2.shape[-1], KERNELS, strides=(pool_size_3_2, pool_size_3_2), padding="same", activation = "relu")(i3)
    u4_3 = tf.keras.layers.Conv2DTranspose(i3.shape[-1], KERNELS, strides=(pool_size_4_3, pool_size_4_3), padding="same", activation = "relu")(i4)
    u5_4 = tf.keras.layers.Conv2DTranspose(i4.shape[-1], KERNELS, strides=(pool_size_5_4, pool_size_5_4), padding="same", activation = "relu")(i5)
    c1 = tf.keras.layers.Add()([i1, u2_1])
    c2 = tf.keras.layers.Add()([i2, u3_2])
    c3 = tf.keras.layers.Add()([i3, u4_3])
    c4 = tf.keras.layers.Add()([i4, u5_4])
    c1 = conv_block(c1, c1.shape[-1], 1, KERNELS, activation = 'relu')
    c2 = conv_block(c2, c2.shape[-1], 1, KERNELS, activation = 'relu')
    c3 = conv_block(c3, c3.shape[-1], 1, KERNELS, activation = 'relu')
    c4 = conv_block(c4, c4.shape[-1], 1, KERNELS, activation = 'relu')
    d2 = tf.keras.layers.Add()([i2, c2])
    d3 = tf.keras.layers.Add()([i3, c3])
    d4 = tf.keras.layers.Add()([i4, c4])
    if bi:
        u2 = conv_block(c1, i2.shape[-1], pool_size_2_1, KERNELS, activation = 'relu')
        u3 = conv_block(c2, i3.shape[-1], pool_size_3_2, KERNELS, activation = 'relu')
        u4 = conv_block(c3, i4.shape[-1], pool_size_4_3, KERNELS, activation = 'relu')
        if bi > 1:
            u5 = conv_block(c4, i5.shape[-1], pool_size_5_4, KERNELS, activation = 'relu')
        e2 = tf.keras.layers.Add()([i2, u2])
        e3 = tf.keras.layers.Add()([i3, u3])
        e4 = tf.keras.layers.Add()([i4, u4])
        if bi > 1:
            e5 = tf.keras.layers.Add()([i5, u5])
        o1 = conv_block(c1, c1.shape[-1], 1, KERNELS, activation = end_activation)
        o2 = conv_block(e2, e2.shape[-1], 1, KERNELS, activation = end_activation)
        o3 = conv_block(e3, e3.shape[-1], 1, KERNELS, activation = end_activation)
        o4 = conv_block(e4, e4.shape[-1], 1, KERNELS, activation = end_activation)
        if bi > 1:
            o5 = conv_block(e5, e5.shape[-1], 1, KERNELS, activation = end_activation)
        else:
            o5 = conv_block(i5, i5.shape[-1], 1, KERNELS, activation = end_activation)
    else:
        o1 = conv_block(c1, c1.shape[-1], 1, KERNELS, activation = end_activation)
        o2 = conv_block(d2, d2.shape[-1], 1, KERNELS, activation = end_activation)
        o3 = conv_block(d3, d3.shape[-1], 1, KERNELS, activation = end_activation)
        o4 = conv_block(i4, i4.shape[-1], 1, KERNELS, activation = end_activation)
        o5 = conv_block(i5, i5.shape[-1], 1, KERNELS, activation = end_activation)
    o1 = tf.keras.layers.Dropout(dropout)(o1)
    o2 = tf.keras.layers.Dropout(dropout)(o2)
    o3 = tf.keras.layers.Dropout(dropout)(o3)
    o4 = tf.keras.layers.Dropout(dropout)(o4)
    o5 = tf.keras.layers.Dropout(dropout)(o5)
    return o1, o2, o3, o4, o5
