"""
Mask R-CNN
The main Mask R-CNN model implementation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM

from mrcnn import utils

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion

assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')


############################################################
#  Utility Functions
############################################################

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints its shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(), array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("", ""))
        text += "  {}".format(array.dtype)
    print(text)


class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place to make changes if needed.

    Batch normalization has a negative effect on training if batches are small,
    so this layer is often frozen (via setting in Config class) and functions as linear layer.
    """

    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)


def compute_backbone_shapes(config, image_shape):
    """Computes the width and height of each stage of the backbone network.

    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)

    # Currently supports ResNet only
    assert config.BACKBONE in ["resnet50", "resnet101"]
    # 平：BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    # matb.ceil() 为向上取整。
    return np.array([[int(math.ceil(image_shape[0] / stride)),
                      int(math.ceil(image_shape[1] / stride))] for stride in config.BACKBONE_STRIDES])


############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def identity_block(input_tensor, kernel_size, filters, stage, block, use_bias=True, train_bn=True):
    """The identity_block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), use_bias=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                                           '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
    """Build a ResNet graph.

    architecture: Can be resnet50 or resnet101
    stage5: Boolean. If False, stage5 of the network is not created
    train_bn: Boolean. Train or freeze Batch Norm layers
    """
    assert architecture in ["resnet50", "resnet101"]

    # KL.Conv2D() 默认 padding='valid'，strides=(1,1)

    # Stage 1
    # 平：设输入的 image 尺寸为 (H,W)，如 (1024, 1024)
    x = KL.ZeroPadding2D((3, 3))(input_image)    # 平: (h+3*2, w+3*2) => (1030, 1030)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    # 平: ((h-7)/2+1, (w-7)/2+1) => 向下取整为 (512, 512)，相当于 (H/2. W/2)。
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # 平：((h-3)/2+1, (w-3)/2+1) => (255, 255)，相当于 (H/4. W/4)

    # Stage 2
    # (平：kernel_size=3, filters=[64, 64, 256])
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)  # 平：(h, w) => (h, w)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)              # 平：(h, w) => (h, w)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
    # 平：Stage 2 只改变其输入的 channel 数，不改变其尺寸。(h, w) => (h, w)，即 (255, 255)，相当于 (H/4. W/4)

    # Stage 3
    # (平：kernel_size=3, filters=[128, 128, 512])
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)      # 平：(h, w) => ((h-1)/2+1, w同理)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)  # 平：(h, w) => (h, w)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
    # 平：Stage 3 除了改变其输入的 channel 数，还改变其尺寸。(h, w) => (h/2, w/2)，即 (127, 127)，相当于 (H/8. W/8)

    # Stage 4
    # (平：kernel_size=3, filters=[256, 256, 1024])
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)      # 平：(h, w) => ((h-1)/2+1, w同理)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]    # 平：architecture="resnet101"
    for i in range(block_count):  # 平：对于 architecture="resnet101"， 有 22 个 identity_block
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)  # 平：(h, w) => (h, w)
    C4 = x
    # 平：Stage 4 除了改变其输入的 channel 数，也改变其尺寸。(h, w) => (h/2, w/2)，即 (64, 64)，相当于 (H/16. W/16)

    # Stage 5
    # (平：kernel_size=3, filters=[256, 256, 1024])
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)  # 平：(h, w) => ((h-1)/2+1, w同理)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)          # 平：(h, w) => (h, w)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None
    # 平：Stage 5 除了改变其输入的 channel 数，也改变其尺寸。(h, w) => (h/2, w/2)，即 (32, 32)，相当于 (H/32. W/32)

    return [C1, C2, C3, C4, C5]


############################################################
#  Proposal Layer
############################################################

def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)] boxes to update
    deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result


def clip_boxes_graph(boxes, window):
    """
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] in the form y1, x1, y2, x2
    """
    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)      # 平：window 可理解为原始 image 所在的范围 (无论图片有没有经过缩放)。
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)

    # Clip
    # 平：假设经过 缩放 和 normalized，则原 image 的 window (0, 0, 1, 1)
    # Clip 的一个原则是：boxes 的 y1, x1, y2, x2 都要在 clip 后满足：
    #   y1 ≥ wy1 = 0; y2 ≥ wy1 = 0;
    #   x1 ≥ wx1 = 0; x2 ≥ wx1 = 0;
    # Clip 的另一个原则是：boxes 的 y1, x1, y2, x2 都要在 clip 后满足：
    #   y1 ≤ wy2 = 1; y2 ≤ wy2 = 1;
    #   x1 ≤ wx2 = 1; x2 ≤ wx2 = 1;
    # 即：
    # boxes 的所有坐标都要在 image 所在的范围内。
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)  # 平：低坐标与低坐标比较，对于 y1 和 wy1 两者，肯定取最大的。
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)  # 平：对于 x1 和 wx1 两者，肯定取最大的。
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


class ProposalLayer(KE.Layer):
    """Receives anchor scores and selects a subset to pass as proposals to the second stage.
    Filtering is done based on anchor scores and non-max suppression to remove overlaps.
    It also applies bounding box refinement deltas to anchors.

    Inputs:
        rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """
    # 平：
    # 初始化参数 proposal_count 的值为 以下两个之一： (ROIs kept after non-maximum suppression (training and inference))
    # config.POST_NMS_ROIS_TRAINING = 2000
    # config.POST_NMS_ROIS_INFERENCE = 1000
    # 初始化参数 nms_threshold 的值为 config.RPN_NMS_THRESHOLD = 0.7

    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def call(self, inputs):
        # 平：inputs = [rpn_class, rpn_bbox, anchors]
        # 平：rpn_class: (batch_size, h*w*3, 2)
        # 平：rpn_bbox:  (batch_size, h*w*3, 4)
        # 平：anchors:   (batch_size, N, 4)

        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]

        # Box deltas [batch, num_rois, 4]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
        # 平：RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

        # Anchors. 平：(batch_size, N, 4)
        anchors = inputs[2]

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        # 平：config.PRE_NMS_LIMIT = 6000。含义为：ROIs kept after tf.nn.top_k and before non-maximum suppression.
        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True, name="top_anchors").indices
        scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y), self.config.IMAGES_PER_GPU)
        deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y), self.config.IMAGES_PER_GPU)
        pre_nms_anchors = utils.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x), self.config.IMAGES_PER_GPU,
                                            names=["pre_nms_anchors"])
        # 平：utils.batch_slice() 的作用是：将第一个参数 (如 [scores, ix]) 作为函数的输入
        #    应用到第二个参数 (如 lambda x, y: tf.gather(x, y)) 所表示的函数中。
        #    只不过因为 batch_slice() 第一个参数(列表类型) 里的每一个元素都是 batch, 应用到其第二个参数时，是分别取第一个参数里每个元素里
        #    batch 单位为 1 的 slice 来输入。最后再将所有 batch 单位为 1 的 slice 的输入所产生的输出，重新分门别类合并为各个 batch。

        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, y2, x2)]
        boxes = utils.batch_slice([pre_nms_anchors, deltas], lambda x, y: apply_box_deltas_graph(x, y),
                                  self.config.IMAGES_PER_GPU, names=["refined_anchors"])

        # Clip to image boundaries. Since we're in normalized coordinates,
        # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
        # 平：函数clip_boxes_graph() 的作用是：clip 之后的 boxes 的所有坐标都要在 image window 所在的范围内。
        #    因为原来的 boxes (由 anchors 通过上述的 tf.nn.top_k 得到) 有一些框柱的范围是超出 原image 的。
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = utils.batch_slice(boxes, lambda x: clip_boxes_graph(x, window), self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors_clipped"])

        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy for small objects, so we're skipping it.

        # Non-max suppression
        # 平：可以参考 utils.py 中的 non_max_suppression(boxes, scores, threshold) 函数来理解。
        # self.nms_threshold = config.RPN_NMS_THRESHOLD = 0.7
        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(boxes, scores, self.proposal_count, self.nms_threshold,
                                                   name="rpn_non_max_suppression")
            proposals = tf.gather(boxes, indices)  # 平：proposals.shape = (n, 4)，即其秩为 2。
            # Pad if needed
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            # 平：由前面 proposals.shape = (n, 4)，即其秩为 2，
            #    记 p = [(0, padding), (0, 0)]，
            #    即对于 proposals 第一个维度(秩)，即行的前面 pad 上 0 行；对于 proposals 第一个维度(秩)，即行的后面， pad 上 padding 行。
            #      对于 proposals 第二个维度(秩)，即列的前面 pad 上 0 列；对于 proposals 第二个维度(秩)，即列的后面， pad 上 0 列。
            return proposals

        proposals = utils.batch_slice([boxes, scores], nms, self.config.IMAGES_PER_GPU)
        return proposals  # 平：Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)] (在 image space 的坐标)

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)


############################################################
#  ROIAlign Layer
############################################################

def log2_graph(x):
    """Implementation of Log2. TF doesn't have a native implementation."""
    return tf.log(x) / tf.log(2.0)


class PyramidROIAlign(KE.Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized coordinates.
        Possibly padded with zeros if not enough boxes to fill the array.
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - feature_maps: List of feature maps from different levels of the pyramid. [P2, P3, P4, P5].
                    Each is [batch, height, width, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
    The width and height are those specific in the pool_shape in the layer constructor.
    """
    # 平：参数 pool_shape 通常取自 config.POOL_SIZE = 7，即通常为 [7, 7]
    # 平：feature_maps [P2, P3, P4, P5] 的 channels 均为 config.TOP_DOWN_PYRAMID_SIZE = 256

    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]

        # Image meta
        # Holds details about the image. See compose_image_meta()
        image_meta = inputs[1]

        # Feature Maps. List of feature maps from different level of the feature pyramid. [P2, P3, P4, P5].
        # Each is [batch, height, width, channels]
        # 平：[P2, P3, P4, P5] 的 channels 均为 config.TOP_DOWN_PYRAMID_SIZE = 256
        feature_maps = inputs[2:]

        # Assign each ROI to a level in the pyramid based on the ROI area.
        # 平：注意！该 class 定义开头的说明文档提到：
        # boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized coordinates.
        # 所以 boxes 这里的坐标都是 normalized coordinates。
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1  # 平：[batch, num_boxes, 1]
        w = x2 - x1  # 平：[batch, num_boxes, 1]
        # Use shape of first image. Images in a batch must have the same size.
        # 平：compose_image_meta() 中参数 image_shape 的说明：[H, W, C] after resizing and padding。
        # 并且注意！这里 image_shape 的值是没有 normalized 的。
        image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]

        # Equation 1 in the Feature Pyramid Networks paper.
        # Account for the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        # 平：“in pixels” 说明了 224x224 的 ROI 是在 image space 的表示。
        #
        # 平：(为了简化表示，下面这段注释涉及的 shape 不 考虑 batch 和 channels 这两个维度。)
        # P4 的 shape 为 原image 的 1/(2^4)， 记 原image 的 shape 为 (H, W)，则 P4 的 shape 为 (H/(2^4), W/(2^4))。
        # 所以，在 image space 中 224x224 的 ROI，maps to P4 为 (14, 14)。
        #
        # 因 config.RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)，即各层
        # backbone feature shape ([C2, C3, C4, C5]，它对应 [P2, P3, P4, P5]) 里产生的 anchors 在 image space 的面积的开根号。
        # 也是 anchors 的 ratio=1 (即 anchors 是正方形) 时的边长。
        # 又因 224 接近于 256，它对应的应该是 P4。
        # [C2, C3, C4, C5] (它对应 [P2, P3, P4, P5]) 的 shape 与 原image 的 shape (记为：(H, W)) 的关系分别为：
        # C2 / P2: (H/(2^2), W/(2^2))
        # C3 / P3: (H/(2^3), W/(2^3))
        # C4 / P4: (H/(2^4), W/(2^4))
        # C5 / P5: (H/(2^5), W/(2^5))
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        # 平：(为了简化表示，下面这段注释涉及的 shape 不 考虑 batch 和 channels 这两个维度。)
        # log2_graph(x) 为：tf.log(x) / tf.log(2.0)， 即 x 取 2 为底的对数。
        # 参考文章：https://blog.csdn.net/u013066730/article/details/102664978
        # 因为 config.RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512) 中的元素是各层的 anchors 在 image space 的面积的开根号。
        # 所以下面公式使用 224.0 (in pixels) 是因为 以 P4 层的 anchors (也即前面注释提到的 ROI) 实际的面积的开根号作为各层的 anchors
        # 的面积的开根号比较的基准。(224 对应 256 这一层)
        # 即 tf.sqrt(h * w) / tf.sqrt(P4 的面积)。
        # 因为 h, w 的值是以 image_shape 进行了 normalized 的了。224.0 (in pixels) 这个值是 image space 里的值，即未 normalized。
        # 所以，要先将 P4 的 anchors 的面积以 image 的面积进行 normalize。即
        # normalized P4 的面积 = (224/image_shape[0]) * (224/image_shape[1]) = 224*224/image_area
        # 所以 tf.sqrt(h * w) / tf.sqrt(P4 的面积)
        #    = tf.sqrt(h * w) / tf.sqrt(224*224/image_area)
        #    = tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area))
        # 然后再取 2 为底的对数，所以下一步得到的 roi_level 是一个相对值，相对于 P4 这一层的层数的差值。
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        # 平：此步得到的 roi_level 的 shape 为 [batch, boxes.shape[1], 1]，
        # 第二个维度里面的值取值范围为 2~5。boxes: [batch, num_boxes, (y1, x1, y2, x2)]，故 boxes.shape[1] = num_boxes。
        #
        # 4 + tf.cast(tf.round(roi_level), tf.int32) 中 4 表示 P4 这一层。就是将层数相对的差值以 P4 层为基准，转化为绝对的层数序号。
        # 然后 与 2 进行 tf.maximum 以及与 5 进行 tf.minimum 是为了使得 roi_level 在 2~5 的范围内，即 [P2, P3, P4, P5]。
        # 下一
        roi_level = tf.squeeze(roi_level, 2)
        # 平：[batch, boxes.shape[1]]，此处第二个维度里面的值取值范围为 2~5。

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            # 平：ix 的 shape 为 [num_batch_level_boxes, 2]。
            #    num_batch_level_boxes 对于 P2~P5 具体指：num_batch_P2_boxes、num_batch_P3_boxes、num_batch_P4_boxes ...
            #    此处 num_batch_level_boxes 表示整个 batch 中属于某一层的 rois 的个数，第二个维度(值为2)表示该层的 rois 在 boxes 的索引。
            #    因为 roi_level 是二维的([batch, boxes.shape[1]])，所以 ix 的第二个维度的值为2，即是两个数。
            #    第一个数代表 roi_level 第一个维度(batch)的索引值，记为 i，i 的取值范围为 0~(batch-1)，表示 batch 的第 i 个 slice。
            #       因一个 batch 由多张 image 构成，故 batch 的 1 个 slice 即为一张 image。batch 的第 i 个 slice 即为第 i 张 image。
            #    第二个数代表 roi_level 第二个维度(boxes.shape[1])的索引值，记为 j，j 的取值范围为 0~(boxes.shape[1]-1)，表示
            #    batch 的第 i 个 slice (即 batch 中的第 i 张 image) 的所有 boxes 中的第 j 个，也就是序号。
            level_boxes = tf.gather_nd(boxes, ix)
            # 平：level_boxes 指整个 batch 中属于某一层的 rois，
            #    其 shape 为 [num_batch_level_boxes, (y1, x1, y2, x2)]，即 [num_batch_level_boxes， 4]

            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)
            # 平：box_indices 的 shape：[num_batch_level_boxes, ]。
            #    ix[:, 0] 即 ix 的第一列，这个列的含义即为 ix (shape 为 [num_batch_level_boxes, 2]) 的第二个维度的两个数中第一个数的含义，
            #    即：代表 roi_level 第一个维度(batch)的索引值，记为 i，i 的取值范围为 0~(batch-1)，表示 batch 的第 i 张 image。
            #    所以 box_indices 的元素的具体取值为 0~(batch-1)，含义为：对应的 roi (或者说 box) 来自 batch 中的第几张图片。

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)
            # 平：由下可知 box_to_level 为 list 类型。其元素的序号依次与 P2~P5 层对应。
            #    其每个元素都为对应的层的 roi 在整个 batch 的 boxes 中的索引集合，其每个元素的 shape 为 [num_batch_level_boxes, 2]。
            #    box_to_level 最终的 shape：
            #    [[num_batch_P2_boxes, 2], [num_batch_P3_boxes, 2], [num_batch_P4_boxes, 2], [num_batch_P5_boxes, 2]]。

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            # 平：[num_batch_level_boxes, (y1, x1, y2, x2)]，即 [num_batch_level_boxes， 4]
            box_indices = tf.stop_gradient(box_indices)
            # 平：[num_batch_level_boxes, ]，各元素表示对应的 roi 来自 batch 的第几张图片。

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so that we can evaluate
            # either max or average pooling. In fact, interpolating only a single value
            # at each bin center (without pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            # 平：上一行注释 Result 里的 num_boxes 可能表示的是 num_batch_level_boxes ，level 分别对应 P2~P5。
            pooled.append(tf.image.crop_and_resize(feature_maps[i],  # 平：[batch, height, width, channels]，即它也是一个 batch。
                                                   level_boxes,      # 平：[num_batch_level_boxes， 4]
                                                   box_indices,      # 平：指对应 roi 来自 batch 的第几张图片(此为 feature map)。
                                                   self.pool_shape,  # 平：通常取自 config.POOL_SIZE = 7，即通常为 [7, 7]
                                                   method="bilinear"))
            # 平：可知 pooled 为 list 类型。
            #    其元素的序号依次对应 P2~P5 层，即 feature_map[i] 用该层 roi (即 level_boxes 中的对应元素) 进行 crop_and_resize 的
            #    结果的集合，也即 ROIAlign 的结果的集合。
            #    feature_maps=[P2, P3, P4, P5]. Each is [batch, height, width, channels]
            #    level_boxes 源自 boxes。注意 boxes 的坐标是 normalized 的，且是在 image space 的坐标 (以 image 的左上角为坐标原点)。
            #    又因各 feature_maps[i] 是 image 通过各 feature_stride 缩放得到，故各 feature_maps[i] 的原点可认为仍是 image space
            #    的原点。也就是说，boxes 的 normalized 的 space，与各 feature_maps[i] 的 space，都是 image space 缩放后得到的相同
            #    原点的 space。故而 boxes 的 normalized coordinates 的特殊性，使得它也适用于各 feature_maps[i] 的 space，用于在各
            #    feature_maps[i] 的 space 来进行定位。

        # Pack pooled features into one tensor
        # 平：由上一步得到的 pooled 形如：
        #    [[batch * num_P2_boxes, pool_height, pool_width, channels],      # [batch * num_P2_boxes, 7, 7, 256]
        #     [batch * num_P3_boxes, pool_height, pool_width, channels],      # [batch * num_P3_boxes, 7, 7, 256]
        #     [batch * num_P4_boxes, pool_height, pool_width, channels],      # [batch * num_P4_boxes, 7, 7, 256]
        #     [batch * num_P5_boxes, pool_height, pool_width, channels]]      # [batch * num_P5_boxes, 7, 7, 256]
        #    注：
        #      num_batch_P2_boxes   + num_batch_P3_boxes   + num_batch_P4_boxes   + num_batch_P5_boxes
        #    = batch * num_P2_boxes + batch * num_P3_boxes + batch * num_P4_boxes + batch * num_P5_boxes
        #    = batch * num_boxes
        #    num_P2_boxes 表示 一张 image 得到的 P2 里面的 box 数，num_boxes 则表示这张照片得到的 P2~P5 所有层总的 box 数。
        pooled = tf.concat(pooled, axis=0)
        # 平：pooled 最终形如：[batch * num_boxes, 7, 7, 256]

        # Pack box_to_level mapping into one array and add another column representing the order of pooled boxes.
        box_to_level = tf.concat(box_to_level, axis=0)
        # 平：此前 box_to_level 的元素表示各层的 roi 在整个 batch 的 boxes 中的索引集合的 shape。
        #    形如：[[num_batch_P2_boxes, 2], [num_batch_P3_boxes, 2], [num_batch_P4_boxes, 2], [num_batch_P5_boxes, 2]]。
        #
        #    也可表示为：[ix_P2, ix_P3, ix_P4, ix_P5]。由前面知，ix 的 shape 为 [num_batch_level_boxes, 2]。
        #    num_batch_level_boxes 对于 P2~P5 具体指：num_batch_P2_boxes、num_batch_P3_boxes、num_batch_P4_boxes ...
        #    此处 num_batch_level_boxes 表示整个 batch 中属于某一层的 rois 的个数，第二个维度(值为2)表示该层的 rois 在 boxes 的索引。
        #    因为 roi_level 是二维的([batch, boxes.shape[1]])，所以 ix 的第二个维度的值为2，即是两个数。
        #    第一个数代表 roi_level 第一个维度(batch)的索引值，记为 i，i 的取值范围为 0~(batch-1)，表示 batch 的第 i 个 slice。
        #       因一个 batch 由多张 image 构成，故 batch 的 1 个 slice 即为一张 image。batch 的第 i 个 slice 即为第 i 张 image。
        #    第二个数代表 roi_level 第二个维度(boxes.shape[1])的索引值，记为 j，j 的取值范围为 0~(boxes.shape[1]-1)，表示
        #    batch 的第 i 个 slice (即 batch 中的第 i 张 image) 的所有 boxes 中的第 j 个，也就是序号。
        #
        #    axis=0 表示列表 box_to_level 中各元素的 shape 的第一个维度的值相加。
        #    最终形如：[batch * num_boxes, 2]
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)              # 平：形如 [batch * num_boxes, 1]
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range], axis=1)  # 平：形如 [batch * num_boxes, 2+1]
        # axis=1 表示列表 [tf.cast(box_to_level, tf.int32), box_range] 中各元素的 shape 的第二个维度的值相加。

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        # 平：box_to_level[:, 0] 元素的值表示第 i 张 image，即相应的 roi 来自第 i 张 image。
        #    box_to_level[:, 1] 元素的值表示相应的 roi 来自第 i 张 image 的所有 boxes 中的第 j 个，也就是序号。
        # 下面的这句注释摘抄自参考文献：https://blog.csdn.net/u013066730/article/details/102664978
        #  sorting_tensor就是batch中的每个图片的索引拉开差距，第0张图片那就是自己，第1张图片需要+100000，第二张图片需要+200000，以此类推
        #
        # 平：假设 batch_size = 3，即 box_to_level[:, 0] 元素的取值为 0,1,2，并假设 box_to_level[:, 0] = [0, 0, 1, 2, 1, ...]
        #    假设一张 image 里 P2~P5 所有 roi 总数为 10，即 box_to_level[:, 1] 的取值为 range(10)，
        #                                                               并假设 box_to_level[:, 1] = [0, 1, 2, 3, 1, ...]
        #    则 sorting_tensor = [0, 1, 100002, 200003, 100001, ...]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(box_to_level)[0]).indices[::-1]
        # 平：
        # tf.nn.top_k().values 返回：The `k` largest elements along each last dimensional slice，且是降序排列好的。例如：
        #   [200009, 200007, 200001, 200001, ..., 200000, 100009, 100005, 100005, ... , 9, 9, ..., 2, 2, ..., 1, ..., 0]
        #   如此一来，batch 中 出自同一张 image 的 roi 就排列在了一起。
        # tf.nn.top_k().indices 返回：
        #   降序排列好的 "The `k` largest elements along each last dimensional slice" 在 each last dimensional slice 的索引。
        #   故此处得到的 ix 为 sorting_tensor 的 each last dimensional slice 元素按升序排列后在原 sorting_tensor 中
        #   的 each last dimensional slice 索引。
        #
        # 故，这里 ix 实现的是 先将整个 batch 的 roi 的 索引 box_to_level 按照 batch 里面的 image 来排序(升序) 以及在同个 image 里的排序。
        # 故，ix 的值是整个 batch 的 roi 的 索引 box_to_level 排序后的索引。即 ix 的值是“索引的索引”。
        # 因 tf.shape(box_to_level)[0] = batch * num_boxes，故 ix 的 shape 为 [ batch * num_boxes, ]。
        ix = tf.gather(box_to_level[:, 2], ix)
        # 平：box_to_level[:, :2] (即整个 batch 的 roi 的 索引)的顺序(即后来新增的 box_to_level[:, 2]) 对应了 pooled 的顺序。
        # 所以这里的作用是将整个 batch 的 roi 的 索引的顺序 box_to_level[:, 2] 按照 ix 的顺序来排序。
        pooled = tf.gather(pooled, ix)
        # 平：实现 pooled 按照 ix 来排序。也即：先按照 batch 里 image 的顺序来排序，且同一个 image 里 也按照原来的 roi 的顺序来排。
        # pooled 形如：[batch * num_boxes, 7, 7, 256]

        # Re-add the batch dimension
        # 平：
        # boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized coordinates.
        # tf.shape(boxes)[:2] = [batch, num_boxes]
        # tf.shape(pooled)[1:] = [7, 7, 256]，也即 [pool_height, pool_width, channels]。
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        return pooled          # 平：Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1],)


############################################################
#  Detection Target Layer
############################################################

def overlaps_graph(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)]. (平：即 [N, 4])
    """
    # 1. Tile boxes2 and repeat boxes1.
    # This allows us to compare every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeat() so simulate it using tf.tile() and tf.reshape.
    # 平：utils.py 的 compute_overlaps() 函数中是通过 for 循环来实现的。
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),     # 平：[N1, 4]    =>      [N1, 1, 4]
                            [1, 1, tf.shape(boxes2)[0]]),  # 平：tf.tile 的第二个参数  [1, 1, N2]
                    [-1, 4])
    # 平：b1.shape 为 [N1*1, 1*1, 4*N2]   =>  [N1, 1, 4*N2]  tf.reshape  =>  [(1*N2)*N1, 4]。
    #    b1 中每 N2 行 是原来 boxes1 的 1 行 复制 N2 份。
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 平：boxes2: [N2, 4]，tf.tile 的第二个参数 [N1, 1]，故 b2.shape 为 [N2*N1, 4*1] => [N2*N1, 4]
    #    b2 中每 N2 行均是原来的 boxes2。

    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)  # 平：[(1*N2)*N1, 4]  =>  [N2*N1, 4]  tf.split => [N2*N1] * 4
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)  # 平：[N2*N1,     4]                  tf.split => [N2*N1] * 4
    y1 = tf.maximum(b1_y1, b2_y1)  # 平：[N2*N1]
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)  # 平：[N2*N1]

    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)         # 平：[N2*N1]
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection            # 平：[N2*N1]

    # 4. Compute IoU and reshape to [boxes1, boxes2]
    # 平：即 overlaps 的 shape 为 [N1, N2]。
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    """Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas, and masks for each.

    Inputs:
    proposals: [POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [MAX_GT_INSTANCES] int class IDs
    gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
    gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.

    Returns: Target ROIs and corresponding class IDs, bounding box shifts, and masks.
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
    deltas: [TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]
    masks: [TRAIN_ROIS_PER_IMAGE, height, width]. Masks cropped to bbox
           boundaries and resized to neural network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """
    # 平：注意返回值 masks 的说明：
    # [TRAIN_ROIS_PER_IMAGE, h, w]. Masks cropped to bbox boundaries and resized to neural network output size.

    # Assertions
    asserts = [tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals], name="roi_assertion"), ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # Remove zero padding
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    # 平：proposals 是 POST_NMS_ROIS_TRAINING, 可认为它是 rpn_rois，在这个函数内由它最终得到 rois (TRAIN_ROIS_PER_IMAGE)。
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros, name="trim_gt_class_ids")
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2, name="trim_gt_masks")
    # 平：tf.where(non_zeros)[:, 0] 表示取 tf.where() 返回结果(index) 的第一列（第一个维度）。

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
    # crowd_boxes = tf.gather(gt_boxes, crowd_ix)   # 平：为了阅读顺畅，将这一句移动到上面。
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
    gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

    # Compute overlaps matrix [proposals, gt_boxes]
    # 平：即 overlaps 的 shape 为 [len(proposals), len(gt_boxes)]。
    overlaps = overlaps_graph(proposals, gt_boxes)

    # Compute overlaps with crowd boxes [proposals, crowd_boxes]
    # 平：即 crowd_overlaps 的 shape 为 [len(proposals), len(crowd_boxes)]
    crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)  # 平：即每个 proposal 与 crowd_boxes(所有) 的最大的 iou。
    no_crowd_bool = (crowd_iou_max < 0.001)
    # 平：因为 crowd_iou_max < 0.001 是一个条件，它紧跟在赋值号 = 之后，加上 () 括起来更便于代码的阅读和理解。
    #    某个 proposal 与 crowd_boxes 的最大的 iou 小于 0.001，则认为该 proposal 与 crowd_boxes 没有相交。
    #    又因为 crowd_boxes 也是属于原始 gt_boxes 中的，只不过 crowd_boxes 中含有多个 instance。
    #    故某个 proposal 与 crowd_boxes 没有相交可认为该 proposal (roi) 里面没有 instance。

    # Determine positive and negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)          # 平：即每个 proposal (roi) 与 crowd_boxes(所有) 的最大的 iou。
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # Assign positive ROIs to GT boxes.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.cond(tf.greater(tf.shape(positive_overlaps)[1], 0),
                                    true_fn=lambda: tf.argmax(positive_overlaps, axis=1),
                                    false_fn=lambda: tf.cast(tf.constant([]), tf.int64))
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # Compute bbox refinement for positive ROIs
    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    # Assign positive ROIs to GT masks
    # Permute masks to [N, height, width, 1]
    # 平：[height, width, N]  tf.transpose  =>  [N, height, width]  tf.expand_dims   =>  [N, height, width, 1]
    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
    # Pick the right mask for each ROI
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)    # 平：[n, height, width, 1]

    # Compute mask targets
    #
    # 平：
    # Compute mask targets 这部分的代码要做的是将 roi_gt_boxes 和 roi_masks 有相交的那部分 masks 给截取出来。前面这句表述错误。
    # Compute mask targets 这部分的代码要做的是在 roi_masks 中将 roi_gt_boxes 这部分范围给截取出来。期望截取到的部分中含有 masks
    # 的范围越大越好，充当相应的 roi_gt_boxes 进行后续训练的 targets (labels)。
    # 所以要确保 roi_gt_boxes 和 roi_masks 这两者 在同一个坐标系下来进行截取。
    # 但是要注意的是：roi_gt_boxes 和 roi_masks 这两者虽然都是 normalized coordinates，但是二者的坐标系有时候是不一样的：
    # ① roi_gt_boxes 的坐标是 feature maps anchors 映射回 image space 后 normalized 的坐标。
    # ② roi_masks 的坐标则要根据 config.USE_MINI_MASK 是否为 True 而有不同的解释：
    #   根据 utils.minimize_mask() 和 utils.expand_mask() 来看：
    #   ②① 原masks 尺寸和 原images 的尺寸是一样大小的。只不过 原mask 的像素是 bool type，只用于区分出 masks 那部分。
    #   ②② mini masks 的 尺寸则是先截取出相应的 gt_boxes 大小的部分，然后根据相应的 gt_boxes 和 MINI_MASK_SHAPE 之间的关系来进行缩放的。
    #
    #   ②①① 当config.USE_MINI_MASK=False，roi_masks 的坐标是 原masks 的 normalized 坐标,故是 image space 下 normalized 坐标。
    #   ②②① 当config.USE_MINI_MASK=True，roi_masks 的坐标是 mini masks 的 normalized 坐标，是 mini-mask space 下的。
    # 因此，将 roi_gt_boxes 和 roi_masks 有相交的那部分 masks 给截取出来的操作要分上述两种情况区别进行。
    boxes = positive_rois
    if config.USE_MINI_MASK:
        # Transform ROI coordinates from normalized image space to normalized mini-mask space.
        # 平：如果使用了 config.USE_MINI_MASK 模式，则 mask 虽然是 normalized coordinates，但是是在 MINI_MASK_SHAPE space 下的
        #    coordinates，而 rois 和 gt_boxes 是在 image space 下的 normalized coordinates，所以需对 rois coordinates 进行转换。
        y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        # 平：为什么是下面这样子转换，还不是很理解。
        # 答：根据 utils.minimize_mask() 和 utils.expand_mask() 来看，
        #    mini masks 的尺寸则是先截取出相应的 gt_boxes 大小的部分，然后根据相应的 gt_boxes 和 MINI_MASK_SHAPE 之间的关系来进行缩放的。
        #    因为宏观层面的参数 config.USE_MINI_MASK=True，所以要将 roi_gt_boxes 的坐标 从 normalized image space 变换到
        #    normalized mini-mask space. 变换需要分两步进行：平移和缩放。
        #    ① 平移：rois 和 gt_boxes 是以 原image 的左上角为坐标原点，mini masks 是以 gt_boxes 的左上角为坐标原点，所以要将 rois 的
        #       坐标原点变换到 mini masks 的坐标原点，即 rois 的坐标减去 gt_boxes 左上角的坐标。
        #    ② 缩放：rois 的横纵坐标分别除以 gt_h 和 gt_w 即可。
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)
    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32),
                                     boxes,
                                     box_ids,
                                     config.MASK_SHAPE)  # 平：参数 crop_size。 config.MASK_SHAPE = [28, 28]
    # 平：
    # 最终 masks 将作为本函数的返回值。注意函数定义开头处关于返回值 masks 的说明：
    # [TRAIN_ROIS_PER_IMAGE, h, w]. Masks cropped to bbox boundaries and resized to neural network output size.
    # 参数 crop_size=config.MASK_SHAPE 和 默认参数 method="bilinear" 实现了实现了 roi_masks 和 positive_rois 的 ROIAlign。

    # Remove the extra dimension from masks.
    # 平：确保 masks 的 shape 为：[n, h, w]，因为前面有对 masks 进行 tf.transpose 和 tf.expand_dims，变成了 [n, h, w, 1]。
    masks = tf.squeeze(masks, axis=3)

    # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
    # binary cross entropy loss.
    # 平：因为进行 bilinear 插值的时候，有些插入的值是浮点数。
    masks = tf.round(masks)

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    # 平：N：表示 pad 上 negative_rois 同等数量的行数，即 N 行。
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    # 平：P 表示 pad 上 P 行，与 positive_rois 无直接关系，而是指 rois 的个数不够 config.TRAIN_ROIS_PER_IMAGE 时需要 pad 的个数。
    # 平：关于下面 tf.pad()：paddings 参数的每一个元素依次代表 tensor 参数的一个维度。
    rois = tf.pad(rois, [(0, P), (0, 0)])                      # 平：在 rois 的第1个维度 (行) 的后面 pad P 行。
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])  # 平：因为 roi_gt_boxes 的个数只等于 positive_rois 的个数。
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])  # 平：在 roi_gt_class_ids 的第1个维度的后面 pad N+P 行。
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])         # 平：在 deltas 的第1个维度 (行) 的后面 pad N+P 行。
    masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])   # 平：在 masks 的第1个维度 (行) 的后面 pad N+P 行。

    return rois, roi_gt_class_ids, deltas, masks


class DetectionTargetLayer(KE.Layer):
    """Subsamples proposals and generates target box refinement, class_ids, and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type

    Returns: Target ROIs and corresponding class IDs, bounding box shifts, and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width]
                 Masks cropped to bbox boundaries and resized to neural network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, config, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        proposals = inputs[0]
        # 即 target_rois，可视为来自 RPN+ProposalLayer 的 rpn_rois 或者是 另外输出的经 normalized 的 input_rois.
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]

        # Slice the batch and run a graph for each slice
        # TODO: Rename target_bbox to target_deltas for clarity
        names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
        outputs = utils.batch_slice([proposals, gt_class_ids, gt_boxes, gt_masks],
                                    lambda w, x, y, z: detection_targets_graph(w, x, y, z, self.config),
                                    self.config.IMAGES_PER_GPU, names=names)
        return outputs

    def compute_output_shape(self, input_shape):
        return [(None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
                (None, self.config.TRAIN_ROIS_PER_IMAGE),     # class_ids
                (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
                (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.MASK_SHAPE[0],
                self.config.MASK_SHAPE[1])                    # masks
                ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]


############################################################
#  Detection Layer
############################################################

def refine_detections_graph(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in normalized coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [num_detections, (y1, x1, y2, x2, class_id, score)] where
        coordinates are normalized.
    """
    # Class IDs per ROI
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    # Class probability of the top class of each ROI
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)
    # Class-specific bounding box deltas
    deltas_specific = tf.gather_nd(deltas, indices)
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = apply_box_deltas_graph(rois, deltas_specific * config.BBOX_STD_DEV)
    # Clip boxes to image window
    refined_rois = clip_boxes_graph(refined_rois, window)

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep = tf.where(class_ids > 0)[:, 0]
    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois, keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """Apply Non-Maximum Suppression on ROIs of the given class."""
        # Indices of ROIs of the given class
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # Apply NMS
        class_keep = tf.image.non_max_suppression(
            tf.gather(pre_nms_rois, ixs),
            tf.gather(pre_nms_scores, ixs),
            max_output_size=config.DETECTION_MAX_INSTANCES,
            iou_threshold=config.DETECTION_NMS_THRESHOLD)
        # Map indices
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        # Pad with -1 so returned tensors have the same shape
        gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)],
                            mode='CONSTANT', constant_values=-1)
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
        return class_keep

    # 2. Map over class IDs
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                         dtype=tf.int64)
    # 3. Merge results into one list, and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]
    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are normalized.
    detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
    ], axis=1)

    # Pad with zeros if detections < DETECTION_MAX_INSTANCES
    gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections


class DetectionLayer(KE.Layer):
    """Takes classified proposal boxes and their bounding box deltas and returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where coordinates are normalized.
    """

    def __init__(self, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.config = config

    # 平：inputs 为一个 list：[rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta]
    def call(self, inputs):
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        image_meta = inputs[3]

        # Get windows of images in normalized coordinates. Windows are the area in the image that excludes the padding.
        # Use the shape of the first image in the batch to normalize the window
        # because we know that all images get resized to the same size.
        m = parse_image_meta_graph(image_meta)
        image_shape = m['image_shape'][0]
        window = norm_boxes_graph(m['window'], image_shape[:2])

        # Run detection refinement graph on each item in the batch
        detections_batch = utils.batch_slice([rois, mrcnn_class, mrcnn_bbox, window],
                                             lambda x, y, w, z: refine_detections_graph(x, y, w, z, self.config),
                                             self.config.IMAGES_PER_GPU)

        # Reshape output
        # [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] in normalized coordinates
        return tf.reshape(detections_batch, [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6])

    def compute_output_shape(self, input_shape):
        return (None, self.config.DETECTION_MAX_INSTANCES, 6)


############################################################
#  Region Proposal Network (RPN)
############################################################

def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """Builds the computation graph of Region Proposal Network.

    feature_map: backbone features [batch, height, width, depth]
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically,
        1 (anchors for every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be applied to anchors.
    """
    # 平：参数 feature_map 的维度 depth=256，它取自 rpn_feature_maps = [P2, P3, P4, P5, P6]中的任一个。
    # 平：参数 anchors_per_location=3。

    # TODO: check if stride of 2 causes alignment issues if the feature map is not even.
    # Shared convolutional base of the RPN
    # 平：((h-3+2*(3-1)/2)/1+1, w同理) => (h,w)，故 (batch_size, h, w, 256) => (batch_size, h, w, 512)
    shared = KL.Conv2D(512, (3, 3),
                       padding='same', strides=anchor_stride, activation='relu', name='rpn_conv_shared')(feature_map)

    # Anchor Score.
    # [batch, height, width, anchors per location * 2].
    # 平：((h-1+0)/1+1, w同理) => (h, w)，故 (batch_size, h, w, 512) => (batch_size, h, w, 2*3)
    x = KL.Conv2D(2 * anchors_per_location, (1, 1),
                  padding='valid', activation='linear', name='rpn_class_raw')(shared)
    # Reshape to [batch, anchors, 2]  (平：这里的维度 anchors = h * w * 3，即 (batch_size, h*w*3, 2))
    rpn_class_logits = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG.
    rpn_probs = KL.Activation("softmax", name="rpn_class_xxx")(rpn_class_logits)

    # Bounding box refinement.
    # [batch, H, W, anchors per location * depth]
    # where depth is [x, y, log(w), log(h)]
    # 平：((h-1+0)/1+1, w同理) => (h, w)，故 (batch_size, h, w, 512) => (batch_size, h, w, 4*3)
    x = KL.Conv2D(4 * anchors_per_location, (1, 1),
                  padding="valid", activation='linear', name='rpn_bbox_pred')(shared)
    # Reshape to [batch, anchors, 4]  (平：这里的维度 anchors = h * w * 3，即 (batch_size, h*w*3, 4))
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)
    # 平：rpn_bbox: Deltas to be applied to anchors.

    return [rpn_class_logits, rpn_probs, rpn_bbox]


def build_rpn_model(anchor_stride, anchors_per_location, depth):
    """Builds a Keras model of the Region Proposal Network.
    It wraps the RPN graph so it can be used multiple times with shared weights.

    anchor_stride: Controls the density of anchors. Typically,
            1 (anchors for every pixel in the feature map), or 2 (every other pixel).
    anchors_per_location: number of anchors per pixel in the feature map
    depth: Depth of the backbone feature map.

    Returns a Keras Model object. The model outputs, when called, are:
    rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
    rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
    rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be applied to anchors.
    """
    # 平：anchor_stride=1,
    # 平：anchors_per_location = len(config.RPN_ANCHOR_RATIOS = [0.5, 1, 2]) = 3
    # 平：depth = config.TOP_DOWN_PYRAMID_SIZE = 256
    input_feature_map = KL.Input(shape=[None, None, depth], name="input_rpn_feature_map")
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return KM.Model([input_feature_map], outputs, name="rpn_model")


############################################################
#  Feature Pyramid Network Heads
############################################################

def fpn_classifier_graph(rois, feature_maps, image_meta, pool_size, num_classes, train_bn=True, fc_layers_size=1024):
    """Builds the computation graph of the feature pyramid network classifier and regressor heads.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized coordinates.
    feature_maps: List of feature maps from different layers of the pyramid, [P2, P3, P4, P5].
        Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta().
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers
    fc_layers_size: Size of the 2 FC layers

    Returns:
        logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to proposal boxes
    """
    # 平：参数 pool_size = config.POOL_SIZE = 7

    # ROI Pooling
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size], name="roi_align_classifier")([rois, image_meta] + feature_maps)
    # 平：PyramidROIAlign() 的返回值：Pooled regions in the shape: [batch, num_rois, pool_height, pool_width, channels].

    # Two 1024 FC layers (implemented with Conv2D for consistency)
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"), name="mrcnn_class_conv1")(x)
    # 平：参数 fc_layers_size = config.FPN_CLASSIF_FC_LAYERS_SIZE = 1024
    # 平：下面这句话摘自 KL.TimeDistributed() 的说明文档：
    # The input should be at least 3D, and the dimension of index one will be considered to be the temporal dimension.
    # 平：output shape: [batch, num_rois, 1, 1, 1024].

    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # 平：output shape: [batch, num_rois, 1, 1, 1024].

    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (1, 1)), name="mrcnn_class_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # 平：output shape: [batch, num_rois, 1, 1, 1024].

    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2), name="pool_squeeze")(x)
    # 平：shared 的 shape: [batch, num_rois, 1024].

    # Classifier head
    mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes), name='mrcnn_class_logits')(shared)
    mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"), name="mrcnn_class")(mrcnn_class_logits)
    # 平：shape: [batch, num_rois, num_classes].

    # BBox head
    # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
    x = KL.TimeDistributed(KL.Dense(num_classes * 4, activation='linear'), name='mrcnn_bbox_fc')(shared)
    # 平：shape: [batch, num_rois, num_classes * 4].

    # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
    s = K.int_shape(x)
    # 平：K.int_shape(): Returns the shape of tensor or variable as a tuple of int or None entries.
    mrcnn_bbox = KL.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)
    # 平：KL.Reshape():
    #    Output shape: (batch_size,) + target_shape

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph(rois, feature_maps, image_meta, pool_size, num_classes, train_bn=True):
    """Builds the computation graph of the mask head of Feature Pyramid Network.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers

    Returns: Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
    """
    # 平：参数 feature_maps 为 [P2, P3, P4, P5]
    # 平：参数 pool_size 为 config.MASK_POOL_SIZE = 14
    # 平：config.NUM_CLASSES = 1 + 1     # Background + balloon (balloon.py 里重写)

    # ROI Pooling
    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size], name="roi_align_mask")([rois, image_meta] + feature_maps)
    # 平：PyramidROIAlign() 的返回值：Pooled regions in the shape: [batch, num_rois, pool_height, pool_width, channels].
    #    即为 [batch, num_rois, 14, 14, channels].

    # Conv layers
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # 平：shape 为 [batch, num_rois, 14, 14, 256].

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # 平：shape 为 [batch, num_rois, 14, 14, 256].

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv3")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn3')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # 平：shape 为 [batch, num_rois, 14, 14, 256].

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn4')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # 平：shape 为 [batch, num_rois, 14, 14, 256].

    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"), name="mrcnn_mask_deconv")(x)
    # 平：KL.TimeDistributed(): Transposed convolution layer (sometimes called Deconvolution).
    # 平：shape 为 [batch, num_rois, 28, 28, 256].
    x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"), name="mrcnn_mask")(x)
    # 平：shape 为 [batch, num_rois, 28, 28, num_classes].

    return x


############################################################
#  Loss Functions
############################################################

def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss. (平： For bounding-box regression.)
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)  # 平：可在 Fast Rcnn 论文里找到该公式。
    return loss


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss. (平：rpn 分类损失)

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive, -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for BG/FG.
    """
    # 平：rpn_class_logits 由 rpn_graph 返回，是特征图 Reshape to [batch, anchors, 2] 但未经过 softmax 激活(未转化为 probs)的值。

    # Squeeze last dim to simplify
    # 平：rpn_match 元素的取值值为 1、-1、0。
    rpn_match = tf.squeeze(rpn_match, -1)

    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    # 平：anchor_class 元素的取值为 1、0。
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)

    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.where(K.not_equal(rpn_match, 0))

    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # Cross entropy loss
    loss = K.sparse_categorical_crossentropy(target=anchor_class,
                                             output=rpn_class_logits,
                                             from_logits=True)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph. (平：rpn 回归损失。)

    config: the model config object.
    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unused bbox deltas. (target_bbox 就是 GT)
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive, -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    # 平：参数 config 之后另外三个参数为 *x = *[input_rpn_bbox, input_rpn_match, rpn_bbox]
    #    其中，input_rpn_bbox, input_rpn_match 都是由 KL.Input() 定义的输入。
    #    rpn_bbox 是函数 rpn_graph() 的返回值之一。是特征图 Reshape to [batch, anchors, 4]的值。

    # Positive anchors contribute to the loss, but negative and neutral anchors (match value of 0 or -1) don't.
    rpn_match = K.squeeze(rpn_match, -1)
    # 平：rpn_match 形如：[batch, anchors, 1]   => [batch, anchors]
    indices = tf.where(K.equal(rpn_match, 1))

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # Trim target bounding box deltas to the same length as rpn_bbox.
    # 平：batch_counts: 每一个 batch 中 positive anchors (值为1) 的计数值。
    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts, config.IMAGES_PER_GPU)
    # 平：target_bbox: [batch, max positive anchors, 4]
    # 平：batch_pack_graph(x, counts, num_rows) 函数：
    #    Picks different number of values from each row in x depending on the values in counts.

    loss = smooth_l1_loss(target_bbox, rpn_bbox)
    # 平：可在 Fast Rcnn 论文里找到该公式。

    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits, active_class_ids):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    active_class_ids: [batch, num_classes].
        Has a value of 1 for classes that are in the dataset of the image,
        and 0 for classes that are not in the dataset.
    """
    # 平：参数 pred_class_logits 是：mrcnn_class_logits
    # 实际用 target_class_ids 和 pred_class_logits 计算交叉熵损失；active_class_ids 用于消除不在图像的预测类别中的类别的预测损失。

    # During model building, Keras calls this function with target_class_ids of type float32. Unclear why.
    # Cast it to int to get around it.
    target_class_ids = tf.cast(target_class_ids, 'int64')

    # Find predictions of classes that are not in the dataset.
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    # TODO: Update this line to work with batch > 1. Right now it assumes all
    #       images in a batch have the same active_class_ids
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)
    # 平：因为对于 active_class_ids 来说，其元素值不是 1 就是 0 (详见函数定义开头处的说明文档)。
    # 这里 为什么 active_class_ids 要取 [0] ？？？

    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_class_ids, logits=pred_class_logits)

    # Erase losses of predictions of classes that are not in the active classes of the image.
    loss = loss * pred_active

    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
    return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]  (就是 GT 框)
    target_class_ids: [batch, num_rois]. Integer class IDs.     (GT 框对应的类别 ID)
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = K.reshape(target_class_ids, (-1,))                  # 平：[batch * num_rois]
    target_bbox = K.reshape(target_bbox, (-1, 4))                          # 平：[batch * num_rois, 4]
    pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 4))   # 平：[batch * num_rois, num_classes, 4]

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indices.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    # Smooth-L1 Loss
    loss = K.switch(tf.size(target_bbox) > 0,
                    smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.    (平： GT mask )
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.       (平：GT框对应的类别ID)
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor with values from 0 to 1.
    """
    # 参考文献：https://blog.csdn.net/sxlsxl119/article/details/103433078
    # L_mask 是 mask 分支上的损失函数，输出大小为 K*m*m，其编码分辨率为 m*m 的K个二进制mask，即K个类别每个对应一个二进制mask，
    # 对每个像素使用sigmoid 函数，L_mask是平均二进制交叉熵损失。
    # RoI 的groundtruth类别为k，L_mask只定义在第k个mask上，其余的mask属于对它没有影响
    # （也就是说在训练的时候，虽然每个点都会有K个二进制mask，但是只有一个k类mask对损失有贡献，这个k值是分类branch预测出来的）。
    #
    # Mask-RCNN 没有类间竞争，因为其他类别不贡献损失。mask分支对每个类别都有预测，依靠分类层选择输出mask
    # （此时大小应该是m*m，之前预测了一个类别出来，只需要输出该类别对应的mask即可）。
    # 使用FCN是因为(其它的)一般方法是对每个像素使用softmax以及多项交叉熵损失，会出现类间竞争。
    # 但二值交叉熵会使得每一类的 mask 不相互竞争，而不是和其他类别的 mask 比较 。


    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks,
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    loss = K.switch(tf.size(y_true) > 0,
                    K.binary_crossentropy(target=y_true, output=y_pred),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss


############################################################
#  Data Generator
############################################################

def load_image_gt(dataset, config, image_id, augment=False, augmentation=None, use_mini_mask=False):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    augment: (deprecated. Use augmentation instead). If true, apply random
        image augmentation. Currently, only horizontal flipping is offered.
    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
        For example, passing imgaug.augmenters.Fliplr(0.5) flips images right/left 50% of the time.
    use_mini_mask: If False, returns full-size masks that are the same height and width as the original image.
        These can be big, for example 1024x1024x100 (for 100 instances).
        Mini masks are smaller, typically, 224x224 and are generated by extracting the bounding box of the object
        and resizing it to MINI_MASK_SHAPE.

    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those of the image
        unless use_mini_mask is True, in which case they are defined in MINI_MASK_SHAPE.
    """
    # Load image and mask
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    original_shape = image.shape
    image, window, scale, padding, crop = utils.resize_image(image,
                                                             min_dim=config.IMAGE_MIN_DIM,
                                                             min_scale=config.IMAGE_MIN_SCALE,
                                                             max_dim=config.IMAGE_MAX_DIM,
                                                             mode=config.IMAGE_RESIZE_MODE)
    mask = utils.resize_mask(mask, scale, padding, crop)

    # Random horizontal flips.
    # TODO: will be removed in a future update in favor of augmentation
    if augment:
        logging.warning("'augment' is deprecated. Use 'augmentation' instead.")
        if random.randint(0, 1):   # 平：实现了 flips images right/left 50% of the time.
            image = np.fliplr(image)
            mask = np.fliplr(mask)

    # Augmentation
    # This requires the imgaug lib (https://github.com/aleju/imgaug)
    if augmentation:
        import imgaug

        # 平：下面这两句注释值得注意！
        # Augmenters that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always test your augmentation on masks.
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad", "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        # Store shapes before augmentation to compare
        image_shape = image.shape
        mask_shape = mask.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        # Change mask to np.uint8 because imgaug doesn't support np.bool  (平：这句注释值得注意！)
        mask = det.augment_image(mask.astype(np.uint8), hooks=imgaug.HooksImages(activator=hook))
        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        # Change mask back to bool
        mask = mask.astype(np.bool)

    # 平：下面这两句注释值得注意！
    # Note that some boxes might be all zeros if the corresponding mask got cropped out.
    # and here is to filter them out
    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]
    # 平：下面这两句注释值得注意！
    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = utils.extract_bboxes(mask)

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    # Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

    # Image meta data
    image_meta = compose_image_meta(image_id, original_shape, image.shape, window, scale, active_class_ids)

    return image, image_meta, class_ids, bbox, mask


def build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, gt_masks, config):
    """Generate targets for training Stage 2 classifier and mask heads. This is not used in normal training.
    It's useful for debugging or to train the Mask RCNN heads without using the RPN head.

    Inputs:
    rpn_rois: [N, (y1, x1, y2, x2)] proposal boxes.
    gt_class_ids: [instance count] Integer class IDs
    gt_boxes: [instance count, (y1, x1, y2, x2)]      (平：gt_boxes 数组的维度为 [instance count, 4])
    gt_masks: [height, width, instance count] Ground truth masks. Can be full size or mini-masks. (平：注意这最后一句话！)

    Returns:
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]    (平：rois 数组的维度为 [iTRAIN_ROIS_PER_IMAGE, 4])
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    bboxes: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (y, x, log(h), log(w))].
           Class-specific bbox refinements.
    masks: [TRAIN_ROIS_PER_IMAGE, height, width, NUM_CLASSES]. Class specific masks cropped
           to bbox boundaries and resized to neural network output size.
    """
    assert rpn_rois.shape[0] > 0
    assert gt_class_ids.dtype == np.int32, "Expected int but got {}".format(gt_class_ids.dtype)
    assert gt_boxes.dtype == np.int32, "Expected int but got {}".format(gt_boxes.dtype)
    assert gt_masks.dtype == np.bool_, "Expected bool but got {}".format(gt_masks.dtype)

    # It's common to add GT Boxes to ROIs but we don't do that here because
    # according to XinLei Chen's paper, it doesn't help.

    # Trim empty padding in gt_boxes and gt_masks parts
    instance_ids = np.where(gt_class_ids > 0)[0]
    assert instance_ids.shape[0] > 0, "Image must contain instances."
    gt_class_ids = gt_class_ids[instance_ids]
    gt_boxes = gt_boxes[instance_ids]
    gt_masks = gt_masks[:, :, instance_ids]

    # Compute areas of ROIs and ground truth boxes.
    rpn_roi_area = (rpn_rois[:, 2] - rpn_rois[:, 0]) * (rpn_rois[:, 3] - rpn_rois[:, 1])
    gt_box_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

    # Compute overlaps [rpn_rois, gt_boxes]
    # 平：--> [N, gt_instance_count]
    overlaps = np.zeros((rpn_rois.shape[0], gt_boxes.shape[0]))
    for i in range(overlaps.shape[1]):
        gt = gt_boxes[i]
        overlaps[:, i] = utils.compute_iou(gt, rpn_rois, gt_box_area[i], rpn_roi_area)

    # Assign ROIs to GT boxes
    rpn_roi_iou_argmax = np.argmax(overlaps, axis=1)   # 各个 rpn_roi 与 所有 gt_box 的最大的 iou 的索引
    rpn_roi_iou_max = overlaps[np.arange(overlaps.shape[0]), rpn_roi_iou_argmax]  # 各个 rpn_roi 与 所有 gt_box 的最大 iou
    # GT box assigned to each ROI
    rpn_roi_gt_boxes = gt_boxes[rpn_roi_iou_argmax]    # 各个 rpn_roi 与 所有 gt_box 的 iou 最大的那个 gt_box 所形成的组合
    rpn_roi_gt_class_ids = gt_class_ids[rpn_roi_iou_argmax]  # 各 rpn_roi 与 所有 gt_box 的 iou 最大的那个 gt_box 的 class id

    # Positive ROIs are those with >= 0.5 IoU with a GT box.
    fg_ids = np.where(rpn_roi_iou_max > 0.5)[0]

    # Negative ROIs are those with max IoU 0.1-0.5 (hard example mining)
    # TODO: To hard example mine or not to hard example mine, that's the question
    # bg_ids = np.where((rpn_roi_iou_max >= 0.1) & (rpn_roi_iou_max < 0.5))[0]
    bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]

    # Subsample ROIs. Aim for 33% foreground.
    # FG
    fg_roi_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    # 平：TRAIN_ROIS_PER_IMAGE = 200,   ROI_POSITIVE_RATIO = 0.33
    if fg_ids.shape[0] > fg_roi_count:
        keep_fg_ids = np.random.choice(fg_ids, fg_roi_count, replace=False)
    else:    # 平：keep_fg_ids 的数量最多不超过 fg_roi_count
        keep_fg_ids = fg_ids
    # BG
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep_fg_ids.shape[0]
    if bg_ids.shape[0] > remaining:
        keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
    else:
        keep_bg_ids = bg_ids
    # Combine indices of ROIs to keep
    keep = np.concatenate([keep_fg_ids, keep_bg_ids])
    # Need more?
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep.shape[0]
    if remaining > 0:
        # Looks like we don't have enough samples to maintain the desired
        # balance. Reduce requirements and fill in the rest. This is
        # likely different from the Mask RCNN paper.

        # There is a small chance we have neither fg nor bg samples.
        if keep.shape[0] == 0:
            # Pick bg regions with easier IoU threshold
            bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]
            assert bg_ids.shape[0] >= remaining
            keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
            assert keep_bg_ids.shape[0] == remaining
            keep = np.concatenate([keep, keep_bg_ids])
        else:
            # Fill the rest with repeated bg rois.
            keep_extra_ids = np.random.choice(keep_bg_ids, remaining, replace=True)
            keep = np.concatenate([keep, keep_extra_ids])
    assert keep.shape[0] == config.TRAIN_ROIS_PER_IMAGE, \
        "keep doesn't match ROI batch size {}, {}".format(keep.shape[0], config.TRAIN_ROIS_PER_IMAGE)

    # Reset the gt boxes assigned to BG ROIs.
    rpn_roi_gt_boxes[keep_bg_ids, :] = 0
    rpn_roi_gt_class_ids[keep_bg_ids] = 0

    # For each kept ROI, assign a class_id, and for FG ROIs also add bbox refinement.
    rois = rpn_rois[keep]
    roi_gt_boxes = rpn_roi_gt_boxes[keep]
    roi_gt_class_ids = rpn_roi_gt_class_ids[keep]
    roi_gt_assignment = rpn_roi_iou_argmax[keep]

    # Class-aware bbox deltas. [y, x, log(h), log(w)]
    bboxes = np.zeros((config.TRAIN_ROIS_PER_IMAGE, config.NUM_CLASSES, 4), dtype=np.float32)
    # bboxes 的维度为 [config.TRAIN_ROIS_PER_IMAGE, config.NUM_CLASSES, 4]，
    # 即：每一个 rois 都包含有 config.NUM_CLASSES 个 [y, x, log(h), log(w)]。
    # 理解为： 每一个 rois 里，每一个 类都会由一个 [y, x, log(h), log(w)]。
    pos_ids = np.where(roi_gt_class_ids > 0)[0]
    bboxes[pos_ids, roi_gt_class_ids[pos_ids]] = utils.box_refinement(rois[pos_ids], roi_gt_boxes[pos_ids, :4])
    # Normalize bbox refinements
    bboxes /= config.BBOX_STD_DEV

    # Generate class-specific target masks
    masks = np.zeros((config.TRAIN_ROIS_PER_IMAGE, config.MASK_SHAPE[0], config.MASK_SHAPE[1], config.NUM_CLASSES),
                     dtype=np.float32)
    for i in pos_ids:
        class_id = roi_gt_class_ids[i]
        assert class_id > 0, "class id must be greater than 0"
        gt_id = roi_gt_assignment[i]  # 前面：roi_gt_assignment = rpn_roi_iou_argmax[keep]
        class_mask = gt_masks[:, :, gt_id]

        if config.USE_MINI_MASK:  # 说明函数参数 gt_masks 传入的值是经过与 config.USE_MINI_MASK 相对应的处理的。这里进行逆处理。
            # Create a mask placeholder, the size of the image
            placeholder = np.zeros(config.IMAGE_SHAPE[:2], dtype=bool)
            # GT box
            gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[gt_id]
            gt_w = gt_x2 - gt_x1
            gt_h = gt_y2 - gt_y1
            # Resize mini mask to size of GT box
            placeholder[gt_y1:gt_y2, gt_x1:gt_x2] = np.round(utils.resize(class_mask, (gt_h, gt_w))).astype(bool)
            # Place the mini batch in the placeholder
            class_mask = placeholder

        # Pick part of the mask and resize it
        y1, x1, y2, x2 = rois[i].astype(np.int32)
        m = class_mask[y1:y2, x1:x2]
        mask = utils.resize(m, config.MASK_SHAPE)
        masks[i, :, :, class_id] = mask

    return rois, roi_gt_class_ids, bboxes, masks


def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.    (平： N 为 number of anchors)
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
               (平： 由下面可知，N 为 config.RPN_TRAIN_ANCHORS_PER_IMAGE；其它为 anchor 和 bbox 之间的 Δ (deltas)，即偏差。)
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    # 返回值 rpn_match: [N]，是一个一维的 array，用于指示每个 anchor 是属于 1 (含有物体)， -1 (背景)，还是 0 (中立，无法判断)
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)

    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))
    # 平：RPN_TRAIN_ANCHORS_PER_IMAGE = 256   # How many anchors per image to use for RPN training

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude them from training.
    # A crowd box is given a negative class ID. (平：因为 gt boxes 框柱的都是 instances，不会框住背景)
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        # Filter out crowds from ground truth class IDs and boxes (平：注意这里的用词 —— filter out。)
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]

        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)  # 平：(N, ?). N: num of anchors
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)                # crowd_iou_max 是一个 array
        # 平：np.amax: Return the maximum of an array or maximum along an axis.
        no_crowd_bool = (crowd_iou_max < 0.001)
        # no_crowd_bool：anchors 的排位或者 rpn_match 的索引，
        # 表示该 anchor 与 crowd 没有 intersect (或者 intersect 的 iou < 0.001)。
        # 这样以来，该 anchor 在 rpn_match 对应的位置的值将赋值为 -1。(见下面)
        # 因为 gt_boxes 表示框住有物体 (包括 crowd) 的 boxes。若一个 anchor 与 gt_boxes 中 属于 crowd 的那种 box
        # 没有相交或者相交面积非常小，则表示该 anchor 与 gt_boxes 也没有相交或者相交的面积非常小。
        # 则该 anchor 在 rpn_match 对应的位置的值将赋值为 -1，表示该 anchor 为背景。
    else:
        # All anchors don't intersect a crowd
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)
    # 平：(N, ?). N: num of anchors, 行是 anchors, 列是 gt_boxes

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)  # (N, ). N: num of anchors
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]  # 每个 anchor (与 gt_boxes) 的最大 iou
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    # rpn_match: [N]，是一个一维的 array。 某个 anchor 的 rpn_match -1 表示该 anchor 框住的判断为背景
    #
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:, 0]
    # gt_iou_argmax 为每个 gt_box (与 anchor) 的最大 iou 的索引的行坐标。
    # 此处 np.argwhere() 得到的是一个二维的索引，因为 overlaps 是一个二维的数组。
    # [:, 0] 表示取出上述所得索引的第一个维度的所有坐标，即每个 gt_box (与 anchor) 的最大 iou 的索引的行坐标。
    # 因为(不同的)每一个 gt box (即每一列) 与 所有 anchor 的 overlaps 的最大值的那个 anchor (行)，可能是同一个。
    # 即(不同列的)每一列中元素值的最大值可能出现在同一行。
    # 故 gt_iou_argmax 可能会出现重复的值，表示同一个行。
    rpn_match[gt_iou_argmax] = 1
    # 即认为不管 iou 是否达标。都认为每个 gt_box 与所有 anchor 相交最大的那个 anchor，
    # 这个 anchor 里面很有可能含有物体。故也将该 anchor 对应的 rpn_match 的位置赋值为 1。 作为 region proposal。
    #
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE - np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    # 平：前面提及：Don't let positives be more than half the anchors。即
    # 不超过 RPN_TRAIN_ANCHORS_PER_IMAGE = 256 这个数的一半。
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinement() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]
        # 每个 anchor (与 gt_boxes) 的最大 iou 的那个 gt box。
        # 注意：anchor_iou_argmax 元素的个数可能大于 gt_boxes 的个数。

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
        # 因为 gt boxes 是只有在训练阶段才有的，而进行预测时，只能基于由 anchors 得出的 region proposal 来预测 bounding box,
        # 因此，rpn_box 都是基于 (相对于) anchors 的 a_h 和 a_w 来得出。
        rpn_bbox[ix] = [(gt_center_y - a_center_y) / a_h,
                        (gt_center_x - a_center_x) / a_w,
                        np.log(gt_h / a_h),
                        np.log(gt_w / a_w),
                        ]
        # Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV    # 平：RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
        ix += 1

    return rpn_match, rpn_bbox


def generate_random_rois(image_shape, count, gt_class_ids, gt_boxes):
    """Generates ROI proposals similar to what a region proposal network would generate.

    image_shape: [Height, Width, Depth]
    count: Number of ROIs to generate
    gt_class_ids: [N] Integer ground truth class IDs
    gt_boxes: [N, (y1, x1, y2, x2)] Ground truth boxes in pixels.

    Returns: [count, (y1, x1, y2, x2)] ROI boxes in pixels.
    """
    # placeholder
    rois = np.zeros((count, 4), dtype=np.int32)

    # Generate random ROIs around GT boxes (90% of count)
    rois_per_box = int(0.9 * count / gt_boxes.shape[0])
    for i in range(gt_boxes.shape[0]):
        gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[i]
        h = gt_y2 - gt_y1
        w = gt_x2 - gt_x1
        # random boundaries
        # (r_y1, r_x1, r_y2, r_x2) 相当于在 (gt_y1, gt_x1, gt_y2, gt_x2) 的基础上
        # h 方向上两边往外扩大 h, 并注意不要超过 image_shape[0]。
        # w 方向上两边往外扩大 w，并注意不要超过 image_shape[1]。
        r_y1 = max(gt_y1 - h, 0)
        r_y2 = min(gt_y2 + h, image_shape[0])
        r_x1 = max(gt_x1 - w, 0)
        r_x2 = min(gt_x2 + w, image_shape[1])

        # To avoid generating boxes with zero area, we generate double what
        # we need and filter out the extra. If we get fewer valid boxes
        # than we need, we loop and try again.
        while True:
            # rois_per_box * 2 表示 double what we need.
            # 2 表示 x1, x2 这两列或者 y1, y2 这两列的元素。
            y1y2 = np.random.randint(r_y1, r_y2, (rois_per_box * 2, 2))
            x1x2 = np.random.randint(r_x1, r_x2, (rois_per_box * 2, 2))
            # Filter out zero area boxes
            threshold = 1
            y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >= threshold][:rois_per_box]
            x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >= threshold][:rois_per_box]
            if y1y2.shape[0] == rois_per_box and x1x2.shape[0] == rois_per_box:
                break

        # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
        # into x1, y1, x2, y2 order
        x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
        y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
        box_rois = np.hstack([y1, x1, y2, x2])
        rois[rois_per_box * i:rois_per_box * (i + 1)] = box_rois

    # Generate random ROIs anywhere in the image (10% of count)
    remaining_count = count - (rois_per_box * gt_boxes.shape[0])
    # To avoid generating boxes with zero area, we generate double what
    # we need and filter out the extra. If we get fewer valid boxes
    # than we need, we loop and try again.
    while True:
        y1y2 = np.random.randint(0, image_shape[0], (remaining_count * 2, 2))
        x1x2 = np.random.randint(0, image_shape[1], (remaining_count * 2, 2))
        # Filter out zero area boxes
        threshold = 1
        y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >= threshold][:remaining_count]
        x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >= threshold][:remaining_count]
        if y1y2.shape[0] == remaining_count and x1x2.shape[0] == remaining_count:
            break

    # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
    # into x1, y1, x2, y2 order
    x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
    y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
    global_rois = np.hstack([y1, x1, y2, x2])
    rois[-remaining_count:] = global_rois
    return rois


def data_generator(dataset, config, shuffle=True, augment=False, augmentation=None,
                   random_rois=0, batch_size=1, detection_targets=False,
                   no_augmentation_sources=None):
    """A generator that returns images and corresponding target class ids, bounding box deltas, and masks.

    dataset: The Dataset object to pick data from
    config: The model config object
    shuffle: If True, shuffles the samples before every epoch
    augment: (deprecated. Use augmentation instead). If true, apply random
        image augmentation. Currently, only horizontal flipping is offered.
    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
        For example, passing imgaug.augmenters.Fliplr(0.5) flips images right/left 50% of the time.
    random_rois: If > 0 then generate proposals to be used to train the network classifier and mask heads.
        Useful if training the Mask RCNN part without the RPN.
    batch_size: How many images to return in each call.
    detection_targets: If True, generate detection targets (class IDs, bbox deltas, and masks).
        Typically, for debugging or visualizations because in training
        detection targets are generated by DetectionTargetLayer.
    no_augmentation_sources: Optional. List of sources to exclude for augmentation.
        A source is string that identifies a dataset and is defined in the Dataset class.

    Returns a Python generator. Upon calling next() on it, the generator returns two lists, inputs and outputs.
        The contents of the lists differs depending on the received arguments:
    inputs list:
    - images: [batch, H, W, C]
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
    - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
    - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
    - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width are those of the image
        unless use_mini_mask is True, in which case they are defined in MINI_MASK_SHAPE.

    outputs list: Usually empty in regular training. But if detection_targets
        is True then the outputs list contains target class_ids, bbox deltas, and masks.
    """
    b = 0  # batch item index  (平：从本函数最下面看出 b 是每一个 batch_size 里面的索引，即 b 的取值范围为：[0, batch_size-1])
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0
    no_augmentation_sources = no_augmentation_sources or []

    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
    # 平：由 image_shape 基于 config.BACKBONE_STRIDES = [4, 8, 16, 32, 64] 向上取整来得到 backbone_shapes (类型为 np.array)。

    # 平：
    # utils.generate_pyramid_anchors 的返回值为 anchors: [N, (y1, x1, y2, x2)].
    # All generated anchors in one array. Sorted with the same order of the given scales.
    # So, anchors of scale[0] come first, then anchors of scale[1], and so on.
    anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,  # (32, 64, 128, 256, 512)
                                             config.RPN_ANCHOR_RATIOS,  # [0.5, 1, 2]
                                             backbone_shapes,           # feature_shapes
                                             config.BACKBONE_STRIDES,   # [4, 8, 16, 32, 64]
                                             config.RPN_ANCHOR_STRIDE)  # 1

    # Keras requires a generator to run indefinitely.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)    # 平： image_index 初始值为 -1
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]

            # If the image source is not to be augmented pass None as augmentation
            if dataset.image_info[image_id]['source'] in no_augmentation_sources:
                image, image_meta, gt_class_ids, gt_boxes, gt_masks = load_image_gt(dataset,
                                                                                    config,
                                                                                    image_id,
                                                                                    augment=augment, augmentation=None,
                                                                                    use_mini_mask=config.USE_MINI_MASK)
            else:
                image, image_meta, gt_class_ids, gt_boxes, gt_masks = load_image_gt(dataset,
                                                                                    config,
                                                                                    image_id,
                                                                                    augment=augment,
                                                                                    augmentation=augmentation,
                                                                                    use_mini_mask=config.USE_MINI_MASK)

            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            if not np.any(gt_class_ids > 0):
                continue

            # RPN Targets
            rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors, gt_class_ids, gt_boxes, config)

            # Mask R-CNN Targets
            if random_rois:
                rpn_rois = generate_random_rois(image.shape, random_rois, gt_class_ids, gt_boxes)
                if detection_targets:
                    # 以下三行注释是平搬抄自函数 build_detection_targets() 的定义开头的说明文档。
                    # Generate targets for training Stage 2 classifier and mask heads.
                    # This is not used in normal training. It's useful for debugging or to train
                    # the Mask RCNN heads without using the RPN head.
                    rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask = build_detection_targets(rpn_rois,
                                                                                            gt_class_ids,
                                                                                            gt_boxes,
                                                                                            gt_masks,
                                                                                            config)

            # Init batch arrays
            if b == 0:
                batch_image_meta = np.zeros((batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_rpn_match = np.zeros([batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros([batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
                batch_images = np.zeros((batch_size,) + image.shape, dtype=np.float32)
                batch_gt_class_ids = np.zeros((batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_boxes = np.zeros((batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                batch_gt_masks = np.zeros((batch_size, gt_masks.shape[0], gt_masks.shape[1], config.MAX_GT_INSTANCES),
                                          dtype=gt_masks.dtype)
                if random_rois:
                    batch_rpn_rois = np.zeros((batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
                    if detection_targets:
                        batch_rois = np.zeros((batch_size,) + rois.shape, dtype=rois.dtype)
                        batch_mrcnn_class_ids = np.zeros((batch_size,) + mrcnn_class_ids.shape, dtype=mrcnn_class_ids.dtype)
                        batch_mrcnn_bbox = np.zeros((batch_size,) + mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype)
                        batch_mrcnn_mask = np.zeros((batch_size,) + mrcnn_mask.shape, dtype=mrcnn_mask.dtype)

            # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                ids = np.random.choice(np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]

            # Add to batch
            batch_image_meta[b] = image_meta
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            batch_images[b] = mold_image(image.astype(np.float32), config)
            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
            if random_rois:
                batch_rpn_rois[b] = rpn_rois
                if detection_targets:
                    batch_rois[b] = rois
                    batch_mrcnn_class_ids[b] = mrcnn_class_ids
                    batch_mrcnn_bbox[b] = mrcnn_bbox
                    batch_mrcnn_mask[b] = mrcnn_mask
            b += 1

            # Batch full?
            if b >= batch_size:   # 说明 b 是每一个 batch_size 里面的索引，即 b 的取值范围为：[0, batch_size-1]
                inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                          batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
                outputs = []

                if random_rois:
                    inputs.extend([batch_rpn_rois])
                    if detection_targets:
                        inputs.extend([batch_rois])
                        # Keras requires that output and targets have the same number of dimensions
                        batch_mrcnn_class_ids = np.expand_dims(batch_mrcnn_class_ids, -1)
                        outputs.extend([batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask])

                yield inputs, outputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise


############################################################
#  MaskRCNN Class
############################################################

class MaskRCNN():
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        """Build Mask R-CNN architecture.

        input_shape: The shape of the input image.
        mode: Either "training" or "inference". The inputs and outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs (平：下面的 shape 参数都是只对于一张 image 来说的，不是对于 batch images 来说的。)
        # 平：config.IMAGE_SHAPE[2] = IMAGE_CHANNEL_COUNT
        input_image = KL.Input(shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
        input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE], name="input_image_meta")
        if mode == "training":
            # RPN GT
            input_rpn_match = KL.Input(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = KL.Input(shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

            # Detection GT (class IDs, bounding boxes, and masks)
            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = KL.Input(shape=[None], name="input_gt_class_ids", dtype=tf.int32)
            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            input_gt_boxes = KL.Input(shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
            # Normalize coordinates
            gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(x, K.shape(input_image)[1:3]))(input_gt_boxes)
            # 3. GT Masks (zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]
            if config.USE_MINI_MASK:
                input_gt_masks = KL.Input(shape=[config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1], None],
                                          name="input_gt_masks", dtype=bool)
            else:
                input_gt_masks = KL.Input(shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                                          name="input_gt_masks", dtype=bool)
        elif mode == "inference":
            # Anchors in normalized coordinates
            input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        #
        # 平： config.BACKBONE = "resnet101"
        if callable(config.BACKBONE):
            _, C2, C3, C4, C5 = config.BACKBONE(input_image, stage5=True, train_bn=config.TRAIN_BN)
        else:
            _, C2, C3, C4, C5 = resnet_graph(input_image, config.BACKBONE, stage5=True, train_bn=config.TRAIN_BN)
        # Top-down Layers
        # 平：config.TOP_DOWN_PYRAMID_SIZE = 256
        # 平：C5: (H/32, W/32, 2048)       =>  P5: (H/32, W/32, 256)
        # 平：C4: (H/16, W/16, 1024)       =>  P4: (H/16, W/16, 256)
        # 平：C3: (H/8, W/8, 512)          =>  P3: (H/8, W/8, 256)
        # 平：C2: (H/4, W/4, 256)          =>  P2: (H/4, W/4, 256)
        # TODO: add assert to varify feature map sizes match what's in config
        P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
        P4 = KL.Add(name="fpn_p4add")([KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
                                       KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
        P3 = KL.Add(name="fpn_p3add")([KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
                                       KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
        P2 = KL.Add(name="fpn_p2add")([KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
                                       KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)  # 平P2:(H/4, W/4, 256)
        P3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)  # 平P3:(H/8, W/8, 256)
        P4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)  # 平P4:(H/16, W/16, 256)
        P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)  # 平P5:(H/32, W/32, 256)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)                     # 平P6:(H/64, W/64, 256)

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        # Anchors
        if mode == "training":
            anchors = self.get_anchors(config.IMAGE_SHAPE)
            # 平：Returns anchor pyramid (from backbone feature, e.g. P2~P6) for the given image size.
            # [N, (normalized_y1, normalized_x1, normalized_y2, normalized_x2)]。总之就是 [N,4]。

            # Duplicate across the batch dimension because Keras requires it
            # TODO: can this be optimized to avoid duplicating the anchors?
            anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
            # A hack to get around Keras's bad support for constants
            anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
            # 平：为什么上面的输入参数是 input_image，它的目的是什么，这是否与 lambda 函数即 tf.Variable(anchors) 的参数 anchors 有关？
        else:
            anchors = input_anchors    # 平：[N,4]。

        # RPN Model
        # 平：config.RPN_ANCHOR_STRIDE = 1
        # 平：config.RPN_ANCHOR_RATIOS = [0.5, 1, 2]
        # 平：TOP_DOWN_PYRAMID_SIZE = 256 (Size of the top-down layers used to build the feature pyramid)
        #
        # 平：下面的变量 rpn 是 build_rpn_model() 函数通过 KM.Model([input_feature_map], outputs, name="rpn_model")
        #    生成，可通过 rpn([input_feature_map]) 来调用。
        rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE, len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)
        # Loop through pyramid layers
        layer_outputs = []           # list of lists
        for p in rpn_feature_maps:   # 平：rpn_feature_maps = [P2, P3, P4, P5, P6]
            layer_outputs.append(rpn([p]))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        # 平：
        # 对于每个 p in [P2, P3, P4, P5, P6], rpn([p]) 的结果为：[rpn_class_logits, rpn_probs, rpn_bbox]。其中：
        # rpn_class_logits: (batch_size, h*w*3, 2)
        # rpn_probs:        (batch_size, h*w*3, 2)
        # rpn_bbox:         (batch_size, h*w*3, 4)
        # 所以，前面注释里说的 [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]] 意思是：
        # 将每层 p 的输出的 rpn_bbox 单独合并到一起，每层 p 的输出的 rpn_probs 单独合并到一起，诸如此类。
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o)) for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates and zero padded.
        # 平：
        # ROIs kept after non-maximum suppression (training and inference)
        # config.POST_NMS_ROIS_TRAINING = 2000
        # config.POST_NMS_ROIS_INFERENCE = 1000
        # config.RPN_NMS_THRESHOLD = 0.7
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training" else config.POST_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(proposal_count=proposal_count,
                                 nms_threshold=config.RPN_NMS_THRESHOLD,
                                 config=config,
                                 name="ROI")([rpn_class, rpn_bbox, anchors])  # 平：anchors: [batch_size, N, 4]
        # 平：ProposalLayer() 里：Filtering is done based on anchor scores and non-max suppression to remove overlaps.
        #    returns: Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)].
        #    相当于通过 nms (non-max suppression) 对 应用了 rpn_bbox (delta) 的 anchors 进行 过滤筛选。得出 rpn_rois (proposals)。

        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image came from.
            active_class_ids = KL.Lambda(lambda x: parse_image_meta_graph(x)["active_class_ids"])(input_image_meta)

            if not config.USE_RPN_ROIS:  # 平：config.USE_RPN_ROIS = True
                # Ignore predicted ROIs and use ROIs provided as an input.
                input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4], name="input_roi", dtype=np.int32)
                # Normalize coordinates (平：也即 target_rois 是经过 normalized 的 input_rois。)
                target_rois = KL.Lambda(lambda x: norm_boxes_graph(x, K.shape(input_image)[1:3]))(input_rois)
            else:
                target_rois = rpn_rois   # 也即 target_rois 就是 rpn_rois。

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training.
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero padded.
            # Equally, returned rois and targets are zero padded.
            # 平：下面返回的 target_bbox 是 输入实参 target_rois 与 gt_boxes 匹配为 positive 那部分两者的 deltas。
            rois, target_class_ids, target_bbox, target_mask = DetectionTargetLayer(config, name="proposal_targets") \
                                                         ([target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            # Network Heads
            # TODO: verify that this handles zero padded ROIs
            # 平：
            # 函数 fpn_classifier_graph() 的作用:
            #       Builds the computation graph of the feature pyramid network classifier and regressor heads.
            # mrcnn_feature_maps = [P2, P3, P4, P5]
            # config.POOL_SIZE = 7;    config.MASK_POOL_SIZE = 14
            # config.FPN_CLASSIF_FC_LAYERS_SIZE = 1024  (Size of the fully-connected layers in the classification graph)
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier_graph(rois,
                                                                               mrcnn_feature_maps,
                                                                               input_image_meta,
                                                                               config.POOL_SIZE,
                                                                               config.NUM_CLASSES,
                                                                               train_bn=config.TRAIN_BN,
                                                                               fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)
            # 平：Returns:
            # logits: [batch, num_rois, NUM_CLASSES] classifier logits(before softmax).
            # probs: [batch, num_rois, NUM_CLASSES] classifier probabilities.
            # bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to proposal boxes.

            # 平：mrcnn_feature_maps = [P2, P3, P4, P5]
            # 平：config.MASK_POOL_SIZE = 14
            # 平：config.NUM_CLASSES = 1 + 1     # Background + balloon (balloon.py 里重写)
            # 平：config.TRAIN_BN = False        # Defaulting to False since batch size is often small.
            mrcnn_mask = build_fpn_mask_graph(rois,
                                              mrcnn_feature_maps,
                                              input_image_meta,
                                              config.MASK_POOL_SIZE,
                                              config.NUM_CLASSES,
                                              train_bn=config.TRAIN_BN)
            # 平：Returns shape 为 [batch, num_rois, 28, 28, num_classes].


            # TODO: clean up (use tf.identify if necessary)
            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

            # Losses
            # 平：rpn 分类损失 (RPN anchor classifier loss)
            rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x),
                                       name="rpn_class_loss")([input_rpn_match, rpn_class_logits])
            # 平：rpn 回归损失
            rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(config, *x),
                                      name="rpn_bbox_loss")([input_rpn_bbox, input_rpn_match, rpn_bbox])
            # 平：mrcnn 分类损失
            class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x),
                                   name="mrcnn_class_loss")([target_class_ids, mrcnn_class_logits, active_class_ids])
            # 平：mrcnn 回归损失
            bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x),
                                  name="mrcnn_bbox_loss")([target_bbox, target_class_ids, mrcnn_bbox])
            # 平：mrcnn mask 损失
            mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x),
                                  name="mrcnn_mask_loss")([target_mask, target_class_ids, mrcnn_mask])

            # Model
            inputs = [input_image, input_image_meta,
                      input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)
            outputs = [rpn_class_logits,    # RPN 的输出。【build_rpn_model() 函数，其核心函数为 rpn_graph()】
                       rpn_class,
                       rpn_bbox,
                       mrcnn_class_logits,  # 平：fpn_classifier_graph() 的输出。
                       mrcnn_class,
                       mrcnn_bbox,
                       mrcnn_mask,          # 平：build_fpn_mask_graph() 的输出。
                       rpn_rois,            # 平：ProposalLayer() 的输出
                       output_rois,         # 平：DetectionTargetLayer() 的输出，再经过 KL.Lambda(..., name="output_rois") 。
                       rpn_class_loss,      # 平：一下均为 损失函数。
                       rpn_bbox_loss,
                       class_loss,
                       bbox_loss,
                       mask_loss]
            model = KM.Model(inputs, outputs, name='mask_rcnn')
        else:
            # Network Heads
            # Proposal classifier and BBox regressor heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier_graph(rpn_rois,
                                                                               mrcnn_feature_maps,
                                                                               input_image_meta,
                                                                               config.POOL_SIZE,
                                                                               config.NUM_CLASSES,
                                                                               train_bn=config.TRAIN_BN,
                                                                               fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
            detections = DetectionLayer(config,
                                        name="mrcnn_detection")([rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])
            # 平：DetectionLayer():
            #    Takes classified proposal boxes and their bounding box deltas and returns the final detection boxes.
            #    returns: [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] in normalized coordinates.
            #             即：[config.BATCH_SIZE, config.DETECTION_MAX_INSTANCES, 6]


            # Create masks for detections
            detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
            mrcnn_mask = build_fpn_mask_graph(detection_boxes,
                                              mrcnn_feature_maps,
                                              input_image_meta,
                                              config.MASK_POOL_SIZE,
                                              config.NUM_CLASSES,
                                              train_bn=config.TRAIN_BN)

            model = KM.Model([input_image, input_image_meta, input_anchors],
                             [detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                             name='mask_rcnn')

        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from mrcnn.parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)

        return model

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the model directory.
        Returns:
            The path of the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(self.model_dir))
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the corresponding Keras function with the addition of multi-GPU support
        and the ability to exclude some layers from loading.

        exclude: list of layer names to exclude
        """
        import h5py
        # Conditional import to support versions of Keras before 2.2
        # TODO: remove in about 6 months (end of 2018)
        try:
            from keras.engine import saving
        except ImportError:
            # Keras before 2.2 used the 'topology' namespace.
            from keras.engine import topology as saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)

    def get_imagenet_weights(self):
        """Downloads ImageNet trained weights from Keras.

        Returns path to weights file.
        """
        from keras.utils.data_utils import get_file
        TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/' \
                                 'releases/download/v0.2/' \
                                 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                TF_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='a268eb855778b3df3c7506639542a6af')
        return weights_path

    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and metrics.
        Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = keras.optimizers.SGD(lr=learning_rate,
                                         momentum=momentum,
                                         clipnorm=self.config.GRADIENT_CLIP_NORM)
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = ["rpn_class_loss", "rpn_bbox_loss", "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = (tf.reduce_mean(layer.output, keepdims=True) * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.add_loss(loss)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
                      for w in self.keras_model.trainable_weights if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(optimizer=optimizer, loss=[None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (tf.reduce_mean(layer.output, keepdims=True) * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.metrics_tensors.append(loss)

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainable layer names
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name, layer.__class__.__name__))

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # \path\to\logs\coco20171029T2315\mask_rcnn_coco_0001.h5 (Windows)
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5 (Linux)
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}")

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None):
        """Train the model.

        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done already, so this actually determines
                the epochs to train in total rather than in this particular call.
        layers: Allows selecting which layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
            augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
            flips images right/left 50% of the time. You can pass complex
            augmentations as well. This augmentation applies 50% of the
            time, and when it does it flips images right/left half the time
            and adds a Gaussian blur with a random sigma in range 0 to 5.

                augmentation = imgaug.augmenters.Sometimes(0.5, [imgaug.augmenters.Fliplr(0.5),
                                                                 imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                                                                 ])
        custom_callbacks: Optional. Add custom callbacks to be called
            with the keras fit_generator method. Must be list of type keras.callbacks.
        no_augmentation_sources: Optional. List of sources to exclude for augmentation.
            A source is string that identifies a dataset and is defined in the Dataset class.
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = data_generator(train_dataset,
                                         self.config,
                                         shuffle=True,
                                         augmentation=augmentation,
                                         batch_size=self.config.BATCH_SIZE,
                                         no_augmentation_sources=no_augmentation_sources)
        val_generator = data_generator(val_dataset,
                                       self.config,
                                       shuffle=True,
                                       batch_size=self.config.BATCH_SIZE)

        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = [keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                                 histogram_freq=0, write_graph=True, write_images=False),
                     keras.callbacks.ModelCheckpoint(self.checkpoint_path, verbose=0, save_weights_only=True),
                     ]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.keras_model.fit_generator(train_generator,
                                       initial_epoch=self.epoch,
                                       epochs=epochs,
                                       steps_per_epoch=self.config.STEPS_PER_EPOCH,
                                       callbacks=callbacks,
                                       validation_data=val_generator,
                                       validation_steps=self.config.VALIDATION_STEPS,
                                       max_queue_size=100,
                                       workers=workers,
                                       use_multiprocessing=True,
                                       )
        self.epoch = max(self.epoch, epochs)

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected as an input to the neural network.

        images: List of image matrices [height,width,depth]. Images can have different sizes.

        Returns 3 Numpy matrices:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding, crop = utils.resize_image(image,
                                                                            min_dim=self.config.IMAGE_MIN_DIM,
                                                                            min_scale=self.config.IMAGE_MIN_SCALE,
                                                                            max_dim=self.config.IMAGE_MAX_DIM,
                                                                            mode=self.config.IMAGE_RESIZE_MODE)
            molded_image = mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = compose_image_meta(0, image.shape, molded_image.shape, window, scale,
                                            np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, mrcnn_mask, original_image_shape, image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the application.

        detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
        mrcnn_mask: [N, height, width, num_classes]
        original_image_shape: [H, W, C] Original image shape before resizing
        image_shape: [H, W, C] Shape of the image after resizing and padding
        window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
                image is excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1) if full_masks else np.empty(original_image_shape[:2] + (0,))

        return boxes, class_ids, scores, full_masks

    def detect(self, images, verbose=0):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, \
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        # Anchors
        anchors = self.get_anchors(image_shape)
        # 平：Returns anchor pyramid (from backbone feature, e.g. P2~P6) for the given image size.
        # 平：[N, (normalized_y1, normalized_x1, normalized_y2, normalized_x2)]。总之就是 [N,4]。

        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)
        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _ = self.keras_model.predict([molded_images, image_metas, anchors],
                                                                         verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks = self.unmold_detections(detections[i],
                                                                                            mrcnn_mask[i],
                                                                                            image.shape,
                                                                                            molded_images[i].shape,
                                                                                            windows[i])
            results.append({"rois": final_rois,
                            "class_ids": final_class_ids,
                            "scores": final_scores,
                            "masks": final_masks,
                            })
        return results

    def detect_molded(self, molded_images, image_metas, verbose=0):
        """Runs the detection pipeline, but expect inputs that are molded already.
        Used mostly for debugging and inspecting the model.

        molded_images: List of images loaded using load_image_gt()
        image_metas: image meta data, also returned by load_image_gt()

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(molded_images) == self.config.BATCH_SIZE, "Number of images must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(molded_images)))
            for image in molded_images:
                log("image", image)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, "Images must have the same size"

        # Anchors
        anchors = self.get_anchors(image_shape)
        # 平：Returns anchor pyramid (from backbone feature, e.g. P2~P6) for the given image size.
        # 平：[N, (normalized_y1, normalized_x1, normalized_y2, normalized_x2)]。总之就是 [N,4]。

        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)
        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _ = self.keras_model.predict([molded_images, image_metas, anchors],
                                                                         verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(molded_images):
            window = [0, 0, image.shape[0], image.shape[1]]
            final_rois, final_class_ids, final_scores, final_masks = self.unmold_detections(detections[i],
                                                                                            mrcnn_mask[i],
                                                                                            image.shape,
                                                                                            molded_images[i].shape,
                                                                                            window)
            results.append({"rois": final_rois,
                            "class_ids": final_class_ids,
                            "scores": final_scores,
                            "masks": final_masks,
                            })
        return results

    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        # 平：Returns anchor pyramid (from backbone feature, e.g. P2~P6) for the given image size.

        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        # 平：由 image_shape 基于 config.BACKBONE_STRIDES = [4, 8, 16, 32, 64] 向上取整来得到 backbone_shapes.
        # backbone_shapes 的类型为 np.array。

        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            # 平：
            # utils.generate_pyramid_anchors 的返回值为 anchors: [N, (y1, x1, y2, x2)]。即 [N, 4]。
            # All generated anchors in one array. Sorted with the same order of the given scales.
            # So, anchors of scale[0] come first, then anchors of scale[1], and so on.
            a = utils.generate_pyramid_anchors(self.config.RPN_ANCHOR_SCALES,  # (32, 64, 128, 256, 512)
                                               self.config.RPN_ANCHOR_RATIOS,  # [0.5, 1, 2]
                                               backbone_shapes,                # 平：feature_shapes，形如 P2~P6。
                                               self.config.BACKBONE_STRIDES,   # [4, 8, 16, 32, 64]
                                               self.config.RPN_ANCHOR_STRIDE)  # 1

            # Keep a copy of the latest anchors in pixel coordinates because it's used in inspect_model notebooks.
            # TODO: Remove this after the notebook are refactored to not use it
            self.anchors = a
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:2])
            # self._anchor_cache 为一个字典类型。
            # key 为 tuple(image_shape)。
            # value 为 [N, (normalized_y1, normalized_x1, normalized_y2, normalized_x2)]。总之就是 [N,4]。

        return self._anchor_cache[tuple(image_shape)]

    def ancestor(self, tensor, name, checked=None):
        """Finds the ancestor of a TF tensor in the computation graph.

        tensor: TensorFlow symbolic tensor.
        name: Name of ancestor tensor to find
        checked: For internal use. A list of tensors that were already searched to avoid loops in traversing the graph.
        """
        checked = checked if checked is not None else []
        # Put a limit on how deep we go to avoid very long loops
        if len(checked) > 500:
            return None
        # Convert name to a regex and allow matching a number prefix
        # because Keras adds them automatically
        if isinstance(name, str):
            name = re.compile(name.replace("/", r"(\_\d+)*/"))

        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function digs through the encapsulation
        and returns the layer that holds the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.get_weights():
                layers.append(l)
        return layers

    def run_graph(self, images, outputs, image_metas=None):
        """Runs a sub-set of the computation graph that computes the given outputs.

        image_metas: If provided, the images are assumed to be already molded (i.e. resized, padded, and normalized)

        outputs: List of tuples (name, tensor) to compute. The tensors are
            symbolic TensorFlow tensors and the names are for easy tracking.

        Returns an ordered dict of results. Keys are the names received in the input and values are Numpy arrays.
        """
        model = self.keras_model

        # Organize desired outputs into an ordered dict
        outputs = OrderedDict(outputs)
        for o in outputs.values():
            assert o is not None

        # Build a Keras function to run parts of the computation graph
        inputs = model.inputs
        print("# Build a Keras function to run parts of the computation graph\n")      # 平添加
        print("type(model)", type(model))            # 平添加
        print("model.trainable: ", model.trainable)  # 平添加
        print("model.uses_learning_phase: ", model.uses_learning_phase)                # 平添加
        print("type(model.uses_learning_phase): ", type(model.uses_learning_phase))    # 平添加
        print("K.learning_phase(): ", K.learning_phase())          # 平添加
        print("\ninputs: ", inputs)                                # 平添加
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            print("① 进入到 model.uses_learning_phase ... 分支")    # 平添加
            inputs += [K.learning_phase()]
            print("inputs: ", inputs)                              # 平添加
        kf = K.function(model.inputs, list(outputs.values()))

        # Prepare inputs
        if image_metas is None:
            molded_images, image_metas, _ = self.mold_inputs(images)
        else:
            molded_images = images
        image_shape = molded_images[0].shape
        # Anchors
        anchors = self.get_anchors(image_shape)
        # 平：Returns anchor pyramid (from backbone feature, e.g. P2~P6) for the given image size.
        # 平：[N, (normalized_y1, normalized_x1, normalized_y2, normalized_x2)]。总之就是 [N,4]。

        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
        model_in = [molded_images, image_metas, anchors]

        # Run inference
        print("\n\n# Run inference\n\nmodel.trainable: ", model.trainable)  # 平添加
        print("model.uses_learning_phase: ", model.uses_learning_phase)     # 平添加
        print("K.learning_phase(): ", K.learning_phase())                   # 平添加
        # print("\nmodel_in: ", model_in)
        print("\ntype(model_in): ", type(model_in))      # 平添加
        print("len(model_in): ", len(model_in))          # 平添加
        print("model_in[0].shape: ", model_in[0].shape)  # 平添加
        print("model_in[1].shape: ", model_in[1].shape)  # 平添加
        print("model_in[2].shape: ", model_in[2].shape)  # 平添加
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            print("② 进入到 if model.uses_learning_phase ... 分支")    # 平添加
            model_in.append(0.)
        outputs_np = kf(model_in)
        print("\n\n")                                                 # 平添加

        # Pack the generated Numpy arrays into a a dict and log the results.
        outputs_np = OrderedDict([(k, v)
                                  for k, v in zip(outputs.keys(), outputs_np)])
        for k, v in outputs_np.items():
            log(k, v)
        return outputs_np


############################################################
#  Data Formatting
############################################################

def compose_image_meta(image_id, original_image_shape, image_shape, window, scale, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.
    original_image_shape: [H, W, C] before resizing or padding.
    image_shape: [H, W, C] after resizing and padding
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real image is (excluding the padding)
    scale: The scaling factor applied to the original image (float32)
    active_class_ids: List of class_ids available in the dataset from which the image came.
        Useful if training on images from multiple datasets where not all classes are present in all datasets.
    """
    meta = np.array([image_id] +                  # size=1
                    list(original_image_shape) +  # size=3
                    list(image_shape) +           # size=3
                    list(window) +                # size=4 (y1, x1, y2, x2) in image coordinates
                    [scale] +                     # size=1
                    list(active_class_ids)        # size=num_classes
                    )
    return meta


def parse_image_meta(meta):
    """Parses an array that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed values.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {"image_id": image_id.astype(np.int32),
            "original_image_shape": original_image_shape.astype(np.int32),
            "image_shape": image_shape.astype(np.int32),
            "window": window.astype(np.int32),
            "scale": scale.astype(np.float32),
            "active_class_ids": active_class_ids.astype(np.int32),
            }


def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed tensors.
    """
    # 平：
    # compose_image_meta() 的部分参数说明：
    # original_image_shape: [H, W, C] before resizing or padding.
    # image_shape: [H, W, C] after resizing and padding
    # window: (y1, x1, y2, x2) in pixels. The area of the image where the real image is (excluding the padding)
    # scale: The scaling factor applied to the original image (float32)
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {"image_id": image_id,
            "original_image_shape": original_image_shape,
            "image_shape": image_shape,
            "window": window,
            "scale": scale,
            "active_class_ids": active_class_ids,
            }


def mold_image(images, config):
    """Expects an RGB image (or array of images) and subtracts the mean pixel
    and converts it to float. Expects image colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)


############################################################
#  Miscellaneous Graph Functions
############################################################

def trim_zeros_graph(boxes, name='trim_zeros'):
    """Often boxes are represented with matrices of shape [N, 4] and are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row in x depending on the values in counts.
    """
    # 平：x        为 target_bbox，形如 [batch, max positive anchors, 4]。
    #    counts   为 batch_counts，即每个 batch 中 positive anchors 的计数值。
    #    num_rows 为 config.IMAGES_PER_GPU = 2 for training mode = 1 for others.
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)


def norm_boxes_graph(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.

    boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)


def denorm_boxes_graph(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.

    boxes: [..., (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)