"""
Mask R-CNN
Common utility functions and classes.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import sys
import os
import logging
import math
import random
import numpy as np
import tensorflow as tf
import scipy
import skimage.color
import skimage.io
import skimage.transform
import urllib.request
import shutil
import warnings
from distutils.version import LooseVersion

# URL from which to download the latest COCO trained weights
COCO_MODEL_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"


############################################################
#  Bounding Boxes
############################################################

def extract_bboxes(mask):
    """Compute bounding boxes from masks.

    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.

    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])  # 同一个坐标轴(此处为 Y 轴)，低值取最大。(X 轴同理)
    y2 = np.minimum(box[2], boxes[:, 2])  # 同一个坐标轴(此处为 Y 轴)，高值取最小。(X 轴同理)
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.

    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    # 计算的是 boxes1 中的每个 box 与 boxes2 中的每个 box 的 overlap。
    # 故 overlaps 组织为 第一个维度(行) 是 boxes1 的个数。第二个维度(列) 是 boxes2 的个数。
    # 即上面注释的：[boxes1 count, boxes2 count]
    for i in range(overlaps.shape[1]):  # model.py 中的 overlaps_graph() 函数无需通过 for 循环就能实现。
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.

    masks1, masks2: [Height, Width, instances]
    """

    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps


def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum suppression and returns indices of kept boxes.

    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    # Compute box areas
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    # Get indicies of boxes sorted by scores (highest first)
    # 平：ixs 是按照 scores 元素值降序排列得到的。
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)

        # Compute IoU of the picked box with the rest
        # 平：注意是第 i 个 box 与剩下所有 box 进行 iou 比较
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])

        # Identify boxes with IoU over the threshold.
        # This returns indices into ixs[1:], so add 1 to get indices into ixs.
        # 平：iou 的列的那些 box 是 ixs[0] 依次与 ixs[1:] 计算得到的。
        #    因为 iou 的列 的索引是从 0 开始的，对应回 ixs[1:] 在 ixs 中的位置就要再 +1。
        remove_ixs = np.where(iou > threshold)[0] + 1

        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.

    boxes: [N, (y1, x1, y2, x2)]. Note that (y2, x2) is outside the box.
    deltas: [N, (dy, dx, log(dh), log(dw))]
    """
    boxes = boxes.astype(np.float32)
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= np.exp(deltas[:, 2])
    width *= np.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    return np.stack([y1, x1, y2, x2], axis=1)


def box_refinement_graph(box, gt_box):
    """Compute refinement needed to transform box to gt_box.

    box and gt_box are [N, (y1, x1, y2, x2)]
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.log(gt_height / height)
    dw = tf.log(gt_width / width)

    result = tf.stack([dy, dx, dh, dw], axis=1)
    return result


def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.

    box and gt_box are [N, (y1, x1, y2, x2)]. (y2, x2) is assumed to be outside the box.
    """
    box = box.astype(np.float32)
    gt_box = gt_box.astype(np.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = np.log(gt_height / height)
    dw = np.log(gt_width / width)

    return np.stack([dy, dx, dh, dw], axis=1)


############################################################
#  Dataset
############################################################

class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset you want to use.
    For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({"source": source,
                                "id": class_id,
                                "name": class_name,
                                })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {"id": image_id,
                      "source": source,
                      "path": path,
                      }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.

        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's available online for easy debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].

        Returns:
            masks: A bool array of shape [height, width, instance count] with a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        logging.warning("You are using the default load_mask(), maybe you need to define your own one.")
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids


def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.
    (平：先确定 scale 来 resize image, 再根据 mode 的不同来确定各种模式下 resized image 的 padding 和 window，甚至 crop。)

    min_dim: if provided, resizes the image such that it's smaller dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up before padding.
               max_dim is ignored in this mode. The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based on min_dim and min_scale,
              then picks a random crop of size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might be inserted in the returned image.
        If so, this window is the coordinates of the image part of the full image (excluding the padding).
        The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    # window 即 padding 之后原 image 显示的位置 (左上角和右下角的坐标)。
    # 初始值即为 padding 前，即原 image 左上角右下角坐标。
    window = (0, 0, h, w)
    scale = 1
    # 本函数返回值之一 padding 的格式为：[(top, bottom), (left, right), (0, 0)]
    # 因为 image 为 三维 (H, W, C)，每一个维度都可分为 前后（或：上下、左右）两个方向来 padding。
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":  # none: No resizing. Return the image unchanged. (平添加)
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:   # 平：确保 scale >= 1
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:  # 平：进一步提升 scale 使之为 min_scale。(说明如果 min_scale 提供的话，应 > 1)
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":  # 平：确保在 max_dim 和 “square” 模式下，scale 不会使 image_max * scale 超过 max_dim。
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    # 平：先 resize image。image 的 H 和 W 都按照上面最终确定下来的 scale 进行 resize。
    if scale != 1:
        image = resize(image, (round(h * scale), round(w * scale)), preserve_range=True)

    # Need padding or cropping?
    # 平：再根据 mode 的不同来确定各种模式下的 padding 和 window，甚至 crop。
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad

        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        # padding 作为 np.pad() 的第二个位置参数。如果需要 pad 的对象为一个 rank 1 array，记它为 vector，
        # 且此时 padding 参数是一个 `二元数组` (front_num, rear_num)。
        # 则需要填充的位置为 vector 的首位置再往前填充 front_num 个位置，末尾位置再往后填充 rear_num 个位置。
        # 因为 image 是 三维的 (H, W, C)，所以 padding 也是需要为一个元素为 `二元数组`，长度为 3 的列表。
        # 变量 padding 第三个元素对应 C 这个维度的值为 (0,0), 说明无需对该维度进行填充。
        # 其实这也是可以理解的，因为我们并不需要增加 image 的通道数，而是只需要对每个通道的 (H, W) 这个二维面积从上下(H)和左右(W)进行填充。

        window = (top_pad, left_pad, h + top_pad, w + left_pad)
        # window 即 原 image 处于填充之后的 image 的位置 (左上角和右下角的坐标)。
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop


def resize_mask(mask, scale, padding, crop=None):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    # Suppress warning from scipy 0.13.0, the output shape of zoom() is
    # calculated with round() instead of int()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    if crop is not None:
        y, x, h, w = crop
        mask = mask[y:y + h, x:x + w]
    else:
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask


def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to reduce memory load.
    Mini-masks can be resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    """
    # 平：参数 bbox 匹配了 mask 被 resize to mini_shape 前的尺寸。
    # bbox: [instances, (y1, x1, y2, x2)]
    # mask: [H, W, instances]
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        # Pick slice and cast to bool in case load_mask() returned wrong dtype
        m = mask[:, :, i].astype(bool)
        y1, x1, y2, x2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        # Resize with bilinear interpolation
        m = resize(m, mini_shape)
        mini_mask[:, :, i] = np.around(m).astype(np.bool)
    return mini_mask


def expand_mask(bbox, mini_mask, image_shape):
    """Resizes mini masks back to image size. Reverses the change of minimize_mask().

    See inspect_data.ipynb notebook for more details.
    """
    # 平：参数 bbox 匹配了 mask 被 resize to mini_shape 前的尺寸。
    # 所以通过 bbox 可以将 mini_mask resize back to mask. 同时 image_shape 参数可以用于匹配原 image 的尺寸.
    mask = np.zeros(image_shape[:2] + (mini_mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mini_mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        h = y2 - y1
        w = x2 - x1
        # Resize with bilinear interpolation
        m = resize(m, (h, w))
        mask[y1:y2, x1:x2, i] = np.around(m).astype(np.bool)
    return mask


# TODO: Build and use this function to reduce code duplication
def mold_mask(mask, config):
    pass


def unmold_mask(mask, bbox, image_shape):
    """Converts a mask generated by the neural network to a format similar to its original shape.

    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.

    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.5
    y1, x1, y2, x2 = bbox
    mask = resize(mask, (y2 - y1, x2 - x1))
    mask = np.where(mask >= threshold, 1, 0).astype(np.bool)

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:2], dtype=np.bool)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask


############################################################
#  Anchors
############################################################

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which to generate anchors. (平：即 feature_shape)
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # 平：
    # 据 config.BACKBONE_STRIDES (形参 feature_strides)，由 config.IMAGE_SHAPE 产生 backbone_shapes (形参 feature shapes)
    #
    # scales: 例：每次调用只被赋予 config.RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512) 中的一个元素。
    # ratios: 例：config.RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    # shape: 例：每次调用只被赋予 feature_shapes 的一个元素。以 1024*1024 的图像为例，通过与 feature_strides 相互作用，
    #        得到的 feature_shapes 为 [[256, 256], [128, 128], [ 64, 64], [ 32, 32], [ 16, 16]]。对应 [P2, P3, P4, P5, P6]。
    # feature_stride: 例：每次调用只被赋予 feature_strides = [4, 8, 16, 32, 64] 中的一个元素。
    # anchor_stride: 例：config.RPN_ANCHOR_STRIDE = 1
    #
    # 进一步解读：
    # 因为各层 feature 的 anchor_stride=1。
    # 且 feature_shape=[256, 256] 这一层 feature 对应的 anchor scale 为 32，且有 ratios 为 [0.5, 1, 2] 三种类型。
    # 故这一层 feature 的 anchor 的总个数为 256*256*3。这几个数的实际意思是：
    # 这一层 feature 映射回 原image 上的 anchor 总个数为 256*256*3，且 stride=4，scale=32，有 ratios=[0.5, 1, 2] 三种类型。
    # 即：
    # feature_shape=[256, 256] 的 feature 映射回 原image 上的 anchor 总个数为 256*256*3，且 stride=4，scale=32，有三种 ratio。
    # 同理：
    # feature_shape=[128, 128] 的 feature 映射回 原image 上的 anchor 总个数为 128*128*3，且 stride=8，scale=64，有三种 ratio。
    # feature_shape=[ 64,  64] 的 feature 映射回 原image 上的 anchor 总个数为 64*64*3，且 stride=16，scale=128，有三种 ratio。
    # feature_shape=[ 32,  32] 的 feature 映射回 原image 上的 anchor 总个数为 32*32*3，且 stride=32，scale=256，有三种 ratio。
    # feature_shape=[ 16,  16] 的 feature 映射回 原image 上的 anchor 总个数为 16*16*3，且 stride=64，scale=512，有三种 ratio。

    # Get all combinations of scales and ratios
    # 平：以参数 scales=32， ratios=[0.5, 1, 2] 为例，
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    # 平：此时的 scales 为：[[32],  [32],  [32]]
    #    此时的 ratios 为：[[0.5], [1. ], [2. ]]
    scales = scales.flatten()    # 平：到这一步 scales=[32, 32, 32]。
    ratios = ratios.flatten()    # 平：到这一步还是 ratios=[0.5, 1, 2]。

    # Enumerate heights and widths from scales and ratios (平：从下面结果来看，是 w/h = ratio)
    heights = scales / np.sqrt(ratios)    # [45.254834,   32.,     22.627417]
    widths = scales * np.sqrt(ratios)     # [22.627417,   32.,     45.254834]

    # Enumerate shifts in feature space
    #
    # 平：将 FPN 某一 level 的 anchor 在 feature map 上的偏移 (anchor stride) 变换为 feature map 在 原image 上的偏移 (feature stride)。
    # 平：即将 feature map 上的坐标变换为在 原image 上的 坐标。
    # 因为 anchor_stride 移动到的在 feature map 上的点都是 anchor 的中心点。所以它们也是原 image 上对应的 box 的中心点。
    # 因此，下面的 (shifts_x, shifts_y) 即为 原image 上对应的 box 的中心点。即它们是在 原image 上的坐标值。
    #
    # print("进入 `# Enumerate shifts in feature space`：")    # 平添加
    # print("shape: ", shape)                                 # 平添加
    # print("anchor_stride: ", anchor_stride)                 # 平添加
    # print("feature_stride: ", feature_stride)               # 平添加
    # print()
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    # 平：以参数 shape=[256, 256] 为例。np.arange(0, 256, 1) * 4  ==>  np.arange(0, 1024, 4)
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    # 平：以参数 shape=[256, 256] 为例。np.arange(0, 256, 1) * 4  ==>  np.arange(0, 1024, 4)
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)
    # 平：shifts_x: [[0, 4, 8, ..., 1012, 1016, 1020],   ...,   [0, 4, 8, ... 1012, 1016, 1020]]，维度为 [256, 256]。
    # 平：shifts_y: [[0,    0,    0,    ..., 0,    0,    0],    [4,    4,    4,    ..., 4,    4,    4],
    #               ...,
    #               [1016, 1016, 1016, ..., 1016, 1016, 1016], [1020, 1020, 1020, ..., 1020, 1020, 1020]]。
    #    维度为 [256, 256]。

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    # 平：因为 shifts_x 是二维，所以 np.meshgrid() 里于先对 shifts_x 进行 flatten (256*256=65536)，再进行 meshgrid。
    # 平：box_widths:  [[22.627417,  32.,  45.254834],  ...,  [22.627417,  32.,  45.254834]]，维度为 (65536, 3)。
    # 平：box_centers_x: [[0, 0, 0], [4, 4, 4], ..., [1016, 1016, 1016], [1020, 1020, 1020],
    #                    ...,
    #                    [0, 0, 0], [4, 4, 4], ..., [1016, 1016, 1016], [1020, 1020, 1020],]。维度为 (65536, 3)。
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
    # 平：同样先对 shifts_y 进行 flatten (256*256=65536)，再进行 meshgrid。
    # 平：box_heights:  [[45.254834,  32., 22.627417 ],  ...,  [45.254834,  32., 22.627417 ]]，维度为 (65536, 3)。
    # 平：box_centers_x: [[0, 0, 0], [4, 4, 4], ..., [0, 0, 0], [0, 0, 0],    # 256 个 [0, 0, 0]
    #                    [4, 4, 4], [4, 4, 4], ..., [4, 4, 4], [4, 4, 4],
    #                    ...,
    #                    [1016, 1016, 1016], [1016, 1016, 1016],  ...,  [1016, 1016, 1016], [1016, 1016, 1016],
    #                    [1020, 1020, 1020], [1020, 1020, 1020],  ...,  [1020, 1020, 1020], [1020, 1020, 1020],]。
    #     维度为 (65536, 3)。

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes, box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides, anchor_stride):
    """Generate anchors at different levels of a feature pyramid.
    Each scale is associated with a level of the pyramid, but each ratio is used in all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array.
        Sorted with the same order of the given scales.
        So, anchors of scale[0] come first, then anchors of scale[1], and so on.
    """
    # 平：
    # 据 config.BACKBONE_STRIDES (形参 feature_strides)，由 config.IMAGE_SHAPE 产生 backbone_shapes (形参 feature shapes)
    #
    # scales: config.RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    # ratios: config.RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    # feature_shapes: 以 1024*1024 的图像为例，与 feature_strides 相互作用，得到的 feature_shapes 为
    #                 [[256, 256], [128, 128], [ 64,  64], [ 32,  32], [ 16,  16]]。对应 [P2, P3, P4, P5, P6]。
    # feature_strides: [4, 8, 16, 32, 64]。对应 [P2, P3, P4, P5, P6] 相比于 原image 的缩小倍数。
    # anchor_stride: config.RPN_ANCHOR_STRIDE = 1
    #
    # 进一步解读：
    # 因为各层 feature 的 anchor_stride=1。
    # 故 feature_shape=[256, 256] 这一层 feature 对应的 anchor scale 为 32，且有 ratios 为 [0.5, 1, 2] 三种类型。
    # 这一层 feature 的 anchor 的总个数为 256*256*3。这几个数的实际意思是：
    # 这一层 feature 映射回 原image 上的 anchor 总个数为 256*256*3，且 stride=4，scale=32，有 ratios=[0.5, 1, 2] 三种类型。
    # 即：
    # feature_shape=[256, 256] 的 feature 映射回 原image 上的 anchor 总个数为 256*256*3，且 stride=4，scale=32，有三种 ratio。
    # 同理：
    # feature_shape=[128, 128] 的 feature 映射回 原image 上的 anchor 总个数为 128*128*3，且 stride=8，scale=64，有三种 ratio。
    # feature_shape=[ 64,  64] 的 feature 映射回 原image 上的 anchor 总个数为 64*64*3，且 stride=16，scale=128，有三种 ratio。
    # feature_shape=[ 32,  32] 的 feature 映射回 原image 上的 anchor 总个数为 32*32*3，且 stride=32，scale=256，有三种 ratio。
    # feature_shape=[ 16,  16] 的 feature 映射回 原image 上的 anchor 总个数为 16*16*3，且 stride=64，scale=512，有三种 ratio。


    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i],
                                        ratios,
                                        feature_shapes[i],
                                        feature_strides[i],
                                        anchor_stride)
                       )
    return np.concatenate(anchors, axis=0)


############################################################
#  Miscellaneous
############################################################

def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.

    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]


def compute_matches(gt_boxes, gt_class_ids, gt_masks,
                    pred_boxes, pred_class_ids, pred_scores, pred_masks,
                    iou_threshold=0.5, score_threshold=0.0):
    """Finds matches between prediction and ground truth instances.

    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = trim_zeros(gt_boxes)
    gt_masks = gt_masks[..., :gt_boxes.shape[0]]
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break

    return gt_match, pred_match, overlaps


def compute_ap(gt_boxes, gt_class_ids, gt_masks,
               pred_boxes, pred_class_ids, pred_scores, pred_masks,
               iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Get matches and overlaps
    gt_match, pred_match, overlaps = compute_matches(
        gt_boxes, gt_class_ids, gt_masks,
        pred_boxes, pred_class_ids, pred_scores, pred_masks,
        iou_threshold)

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps


def compute_ap_range(gt_box, gt_class_id, gt_mask,
                     pred_box, pred_class_id, pred_score, pred_mask,
                     iou_thresholds=None, verbose=1):
    """Compute AP over a range or IoU thresholds. Default range is 0.5-0.95."""
    # Default is 0.5 to 0.95 with increments of 0.05
    iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)

    # Compute AP over range of IoU thresholds
    AP = []
    for iou_threshold in iou_thresholds:
        ap, precisions, recalls, overlaps = compute_ap(gt_box, gt_class_id, gt_mask,
                                                       pred_box, pred_class_id, pred_score, pred_mask,
                                                       iou_threshold=iou_threshold)
        if verbose:
            print("AP @{:.2f}:\t {:.3f}".format(iou_threshold, ap))
        AP.append(ap)
    AP = np.array(AP).mean()
    if verbose:
        print("AP @{:.2f}-{:.2f}:\t {:.3f}".format(iou_thresholds[0], iou_thresholds[-1], AP))
    return AP


def compute_recall(pred_boxes, gt_boxes, iou):
    """Compute the recall at the given IoU threshold. It's an indication
    of how many GT boxes were found by the given prediction boxes.

    pred_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    gt_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    """
    # Measure overlaps
    overlaps = compute_overlaps(pred_boxes, gt_boxes)
    iou_max = np.max(overlaps, axis=1)
    iou_argmax = np.argmax(overlaps, axis=1)
    positive_ids = np.where(iou_max >= iou)[0]
    matched_gt_boxes = iou_argmax[positive_ids]

    recall = len(set(matched_gt_boxes)) / gt_boxes.shape[0]
    return recall, positive_ids


# ## Batch Slicing
# Some custom layers support a batch size of 1 only, and require a lot of work to support batches greater than 1.
# This function slices an input tensor across the batch dimension and feeds batches of size 1.
# Effectively, an easy way to support batches > 1 quickly with little code modification.
# In the long run, it's more efficient to modify the code to support large batches and getting rid of this function.
# Consider this a temporary solution
def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given computation graph
    and then combines the results. It allows you to run a graph on a batch of inputs
    even if the graph is written to support one instance only.

    inputs: list of tensors. All must have the same first dimension length.
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        # 平：可认为 inputs 的各个元素的第一个维度为 batch_size。则这里 x[i] 为 inputs 各元素取其 batch 单位为1的切片重新构成的列表。
        # 以 inputs = [scores, ix] 为例。
        #   scores 为 [Batch, num_rois, 1]
        #   ix 为 score 的索引的集合，维度为 [Batch, top_k_num_rois, 1]
        # 则 对于每一步的 i，inputs_slice 为 inputs 各元素取其 batch 单位为1的切片重新构成的列表，即 [ scores[i,:,:], ix[i,:,:] ]

        output_slice = graph_fn(*inputs_slice)
        # 平：由 input_slice (即 batch_size=1 的 slice) 经 graph_fn 得到 output_slice。
        if not isinstance(output_slice, (tuple, list)):  # 平：保证 output_slice 是 list 或 tuple 类型。
            output_slice = [output_slice]
        outputs.append(output_slice)  # 平：outputs 是各 output_slice 列表构成的一个嵌套列表。即 output_slice 是 list of slice。
    # Change outputs from a list of slices where each is a list of outputs
    # to a list of outputs and each has a list of slices
    # 平：即将各 output_slice (由 *outputs 得到) 的各元素按照同类重新组成(通过zip)一个组合，
    #    各个同类的元素组成的这个组合相当于一个该类元素的 batch，并且各类元素的 batch 又组成一个大组合，最后将这个大组合转换为 list 类型。
    #    即 outputs 是一个列表，它的各元素相当于各个不同类的 batch。
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n) for o, n in zip(outputs, names)]
    # 平：将 outputs 各元素的这个 类batch组合 通过  tf.stack 转换成一个真正的 batch。
    if len(result) == 1:
        result = result[0]

    return result


def download_trained_weights(coco_model_path, verbose=1):
    """Download COCO trained weights from Releases.

    coco_model_path: local path of COCO trained weights
    """
    if verbose > 0:
        print("Downloading pretrained model to " + coco_model_path + " ...")
    with urllib.request.urlopen(COCO_MODEL_URL) as resp, open(coco_model_path, 'wb') as out:
        shutil.copyfileobj(resp, out)
    if verbose > 0:
        print("... done downloading pretrained model!")


def norm_boxes(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.

    boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)


def denorm_boxes(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.

    boxes: [N, (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)


def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    """A wrapper for Scikit-Image resize().

    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(image, output_shape,
                                        order=order, mode=mode, cval=cval, clip=clip,
                                        preserve_range=preserve_range, anti_aliasing=anti_aliasing,
                                        anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(image, output_shape,
                                        order=order, mode=mode, cval=cval, clip=clip,
                                        preserve_range=preserve_range)