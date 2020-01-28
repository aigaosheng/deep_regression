#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The file is to set constant variables for MeRadio data precossing 
Author: SHENG GAO
Date: Jan.17, 2019
"""

#--------------------------------
# Setting for aggregating views
#
#if set DEBUG=True, save raw record
DEBUG = False

DNN_CONFIG = {
        'is_log': True,
        'model_type': 'dnn', #'dnn', #'embed'
        'meta_layer_dim': 64, #[64]*2+[32], 256, #512, #set layer size for meta feature
        'layer_stack': [64*2, 64, 32, ], #[64] * 4,
        'is_batch_norm': False, #if True, use batch_normalize
        'dropout_rate': 0., #0.1, #if None, not use dropout. other must be value in 0.0-1.0
        'i_dim': 0,
        'o_dim': 1,
        'learning_rate': 0.005 * 1,
        'n_epoch': 1000,
        'n_batch_size': 32,
        'activation_i': 'relu', #tanh', #'relu', #internal layer
        'activation_o': 'relu', #'relu', #internal layer
        'n_internal_layer': 4, #4, #number of dense internal layers
        'is_continue_train': False, #True: if a model already exists in the model-path fold, start train from the model as intialization. False: train from random intialization 
        'validation_split': 0.1,
        'loss': 'mean_squared_error', #mse', #smapeLoss', #'mean_squared_logarithmic_error', #'mse', #mean_squared_error', #mean_squared_logarithmic_error', #mse',
        'val_metric': 'mean_squared_error', #'mean_squared_error', #'smapeMetric',
        #for meta embedding
        'seq_length': 3, #1, only support 3 or 1
        'embed_dim': 64, #output dimension of embedded dim
        'n_lstm_dim': 64, #cell size of LSTM
    }

#set for checkpoint, if there is best model in current model path, will continue training from this model. empty folder if training from intialization
#CHECKPOINT_MODEL_NAME = '-{epoch:02d}-{val_loss:.4f}.hdf5' #set checkpoint model extened name
CHECKPOINT_MODEL_NAME = '-{epoch:02d}-{val_%s:.4f}.hdf5' % DNN_CONFIG['val_metric'] #set checkpoint model extened name
CHECKPOINT_MODEL_MONITOR = 'val_%s' % DNN_CONFIG['val_metric'] #'val_loss'
#CHECKPOINT_MODEL_MONITOR = DNN_CONFIG['val_metric']
CHECKPOINT_MODEL_VERBOSE = 1
CHECKPOINT_MODEL_SAVE_BEST_ONLY = True
CHECKPOINT_MODEL_MODE = 'min'
CHECKPOINT_MODEL_PERIOD = 1
REMOVE_CHECKPOINT_MODEL = True #if true, remove checkpoint models after best model is chosen and saved

