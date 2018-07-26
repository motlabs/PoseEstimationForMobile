# Copyright 2018 Zihua Zeng (jwkang10@gmail.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===================================================================================
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import os
import time
import numpy as np
import configparser
import dataset

from datetime import datetime

from dataset import get_train_dataset_pipline
from networks import get_network
from dataset_prepare import CocoPose
from dataset_augment import set_network_input_wh, set_network_scale


def get_train_input(batchsize, epoch):
    train_ds = get_train_dataset_pipline(batch_size=batchsize, epoch=epoch, buffer_size=100)
    iter = train_ds.make_one_shot_iterator()
    _ = iter.get_next()
    return _[0], _[1]



class ModelTest(tf.test.TestCase):

    def test_mv2_hourglass_shape(self):

        batchsize   = 1
        epoch_num   = 1
        model       = 'mv2_hourglass'

        input_image, input_heat = get_train_input(batchsize=batchsize,epoch=epoch_num)


        net  = get_network(model, input_image, True)



