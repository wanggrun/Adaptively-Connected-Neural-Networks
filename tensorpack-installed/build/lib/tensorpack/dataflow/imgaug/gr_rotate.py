# -*- coding: UTF-8 -*-
# File: misc.py

import numpy as np
import cv2

# from tensorpack.dataflow.imgaug.base import ImageAugmentor
# from tensorpack.utils import logger
# from tensorpack.utils.argtools import shape2d
# from tensorpack.dataflow.imgaug.transform import ResizeTransform, TransformAugmentorBase

from .base import ImageAugmentor
from ...utils import logger
from ...utils.argtools import shape2d
from .transform import ResizeTransform, TransformAugmentorBase
import math


__all__ = ['GrRotate']


class GrRotate(ImageAugmentor):
    """
    """
    def __init__(self, angle=0):
        """
        """
        super(GrRotate, self).__init__()
        if angle < 0 or angle > 180:
            raise ValueError("Angle should be between [0, 180]!")
        self._init(locals())

    def _get_augment_params(self, img):
        h, w = img.shape[:2]
        do = int(self._rand_range() * float(self.angle) * 2 - float(self.angle))
        return (do, h, w)

    def _augment(self, img, param):
        do, h, w = param
        h_New=int(w*math.fabs(math.sin(math.radians(do)))+h*math.fabs(math.cos(math.radians(do))))
        w_New=int(h*math.fabs(math.sin(math.radians(do)))+w*math.fabs(math.cos(math.radians(do))))

        matRotation=cv2.getRotationMatrix2D((w/2,h/2),do,1)
        matRotation[0,2] +=(w_New-w)/2
        matRotation[1,2] +=(h_New-h)/2
        ret=cv2.warpAffine(img,matRotation,(w_New,h_New),borderValue=(128,128,128))
        ret = cv2.resize(ret, (w, h))

        if img.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret

