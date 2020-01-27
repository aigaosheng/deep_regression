#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This module is to define metrics used in DNN training, e.g. you can train model by optimizing a specific metric

Author: GAO SHENG
Date: Oct.25, 2018


'''
from __future__ import absolute_import

import sys, os

import random
import numpy as np
from keras import backend as K
from keras import metrics

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import settings

def smapeLoss(y_true, y_pred):
    '''
    define smape (Symmetric Mean Absolute Percent Error) for training
    '''
    df = K.abs(y_true - y_pred)
    #add 1.0 / 3.0 to avoid overfitting, and also improve performance
    if settings.SEQ2INST_PARAM['is_log_pv_y']:
        sf = y_true + y_pred + 1.0 #K.epsilon() # 2.0 #K.mean(y_true, axis = -1)
    else:
        sf = y_true + y_pred + 3.0 #K.epsilon() # 2.0 #K.mean(y_true, axis = -1)

    sc = 2.0 * K.mean(df / sf, axis = -1)
    return sc

def smapeMetric(y_true, y_pred):
    '''
    define smape (Symmetric Mean Absolute Percent Error) for evluating model
    '''
    if settings.SEQ2INST_PARAM['is_log_pv_y']:
        y_true = K.exp(y_true) - 1.
        y_pred = K.exp(y_pred) - 1.
    df = K.abs(y_true - y_pred)
    sf = y_true + y_pred + K.epsilon()
    sc = 2.0 * K.mean(df / sf, axis = -1)

    return sc

#define metric names 
METRIC_LIST = {
    'smapeLoss': smapeLoss,
    'smapeMetric': smapeMetric,
    'mean_squared_error': metrics.mean_squared_error,
    'mae': metrics.mean_absolute_error,
    'mape': metrics.mean_absolute_percentage_error,
    'mse': metrics.mean_squared_error
}    