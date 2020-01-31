#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This is to train VV model using DNN

Author: SHENG GAO
Date: Oct.1, 2018


'''
from __future__ import absolute_import

import sys, csv, os

sys.path.append('..')

import datetime
import random
import math
import pickle
#from pyarrow import deserialize_from
import numpy as np
import tensorflow as tf
from keras import backend as K, regularizers
import json
import glob, shutil
import pandas as pd


import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Input, Embedding, LSTM, BatchNormalization, Dropout
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler

#from sklearn.preprocessing import Normalizer
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evalMetric import getMetric
import settings
from dnnMetric import METRIC_LIST

def feat_tune(feat_pipline, x_train):
    feat_model = []
    for f in feat_pipline:
        feat_model.append(f.fit(x_train))
        x_train = feat_model[-1].transform(x_train)
    return feat_model

def feat_transform(feat_model, x_feat):
    for f in feat_model:
        x_feat = f.transform(x_feat)
    return x_feat

is_feature_tuning = True

def fit(i_train_data_file, o_save_model_file, i_dev_data_file):
    '''
    automatic tuning model using dev data set
    Input:
        i_train_data_file: data file including training instances
        i_dev_data_file: data file including deve instances
        o_save_model_file: save best model to the file
        model_method: model method
        topic_type: method to cluster training instances. If None, using generic model
    '''

    #i_train_data_file = '/home/gao/Work/aifintech/data/feat/美元兑人民币.pkl'
    with open(i_train_data_file, 'rb') as fl:
        data_set = pickle.load(fl)

    x_train, y_train = data_set['train'][:2]
    #y_train = y_train[:, 0]
    if settings.DNN_CONFIG['is_log']:
        y_train = np.log(y_train)

    if is_feature_tuning:
        feat_pipline = [StandardScaler(), Normalizer(norm = 'l2')]
        feat_model =  feat_tune(feat_pipline, x_train)
        x_train = feat_transform(feat_model, x_train)
    else:
        feat_model = None
    
    #norm_me = StandardScaler()
    #feat_pipline.append(norm_me)
    #norm_me2 = Normalizer(norm = 'l2')
    #norm_me.fit(x_train)
    #norm_me2.fit(x_train)
    #x_train = norm_me.transform(x_train)
    #x_train = norm_me2.transform(x_train)


    #y_train=np.exp(y_train)
    x_dev, y_dev = data_set['dev'][:2]
    #y_dev = y_dev[:, 0]
    if settings.DNN_CONFIG['is_log']:
        y_dev = np.log(y_dev)

    if is_feature_tuning:
        x_dev = feat_transform(feat_model, x_dev)
    #x_dev = norm_me.transform(x_dev)
    #x_dev = norm_me2.transform(x_dev)

    print('INFO: %d instances loaded from from %s' % (x_train.shape[0], i_train_data_file))
 
    #train model
    #if o_save_model_file:
    #    o_save_model_file_name = o_save_model + '.' + 'generic'
    #else:
    #    raise Exception('Warning: must set save model file names')
    #add
    model_name = 'me_model.pkl'
    save_model = '/home/gao/Work/deep_regression/exp/model_fin'


    nnConfig = settings.DNN_CONFIG
    nnConfig['i_dim'] = x_train.shape[1]

    tf.random.set_seed(1234)
    #session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
    #sess = tf.Session(graph=tf.get_default_graph())#, config=session_conf)
    #K.set_session(sess)
    #
    gDnnModel = defineDnnModel(nnConfig, model_name)

    sgd = SGD(lr = nnConfig['learning_rate'], decay = 1e-6, momentum = 0.9, nesterov=True)

    if nnConfig['val_metric'] not in METRIC_LIST:
        raise ValueError('Warning: val_metric must be %s' % ','.join(METRIC_LIST.keys()))
    addUserDefinedMetric()
    #if save_model exist, continue training from the model
    if os.path.exists(save_model) and nnConfig['is_continue_train']:
        print('Continue training from %s' % save_model)
        #gDnnModel.load_weights(save_model)
        gDnnModel = load_model(save_model)
    #
    gDnnModel.compile(optimizer = sgd, loss = METRIC_LIST[nnConfig['loss']], metrics = [METRIC_LIST[nnConfig['val_metric']]])    

    #add checkpoint
    filepath = save_model + settings.CHECKPOINT_MODEL_NAME
    checkpoint = ModelCheckpoint(filepath, monitor = settings.CHECKPOINT_MODEL_MONITOR, verbose = settings.CHECKPOINT_MODEL_VERBOSE, save_best_only = settings.CHECKPOINT_MODEL_SAVE_BEST_ONLY, mode = settings.CHECKPOINT_MODEL_MODE, period = settings.CHECKPOINT_MODEL_PERIOD)
    #checkpoint = ModelCheckpoint(filepath, verbose = settings.CHECKPOINT_MODEL_VERBOSE, save_best_only = settings.CHECKPOINT_MODEL_SAVE_BEST_ONLY, mode = settings.CHECKPOINT_MODEL_MODE, period = settings.CHECKPOINT_MODEL_PERIOD)

    callbacks_list = [checkpoint]

    # Fit the model
    if x_dev is not None:
        history = gDnnModel.fit(x_train, y_train, epochs = nnConfig['n_epoch'], batch_size = nnConfig['n_batch_size'], validation_data = (x_dev, y_dev), callbacks = callbacks_list)
    else:
        history = gDnnModel.fit(x_train, y_train, epochs = nnConfig['n_epoch'], batch_size = nnConfig['n_batch_size'], validation_split = nnConfig['validation_split'], callbacks = callbacks_list)
    #print(history.history)

    bestModel = getBestModelFromHistory(save_model, history.history['val_' + nnConfig['val_metric']])

    #save training log to file
    '''tempfile_name = 'train_' + str(random.randint(0,100)) + '.log'
    with open(tempfile_name, 'wt') as oTempLog:
        json.dump(history.history, oTempLog, indent=2)
    '''
    return bestModel, feat_model, None# history.history['val_' + nnConfig['val_metric']]


def addUserDefinedMetric():
    if not hasattr(keras.metrics, settings.DNN_CONFIG['val_metric']):
        setattr(keras.metrics, settings.DNN_CONFIG['val_metric'], METRIC_LIST[settings.DNN_CONFIG['val_metric']])
    if not hasattr(keras.losses, settings.DNN_CONFIG['loss']):
        setattr(keras.losses, settings.DNN_CONFIG['loss'], METRIC_LIST[settings.DNN_CONFIG['loss']])
    

def defineDnnModel(nnConfig, model_name):
    input_vec = Input(shape = (nnConfig['i_dim'],), name=model_name+'_i')
    i_feat = input_vec
    #for k in range(nnConfig['n_internal_layer']):
    for k, layer_size in enumerate(nnConfig['layer_stack']):
        if nnConfig['is_batch_norm']:
            i_feat = BatchNormalization()(i_feat)
        internal_layer = Dense(layer_size, activation = nnConfig['activation_i'], kernel_initializer='glorot_uniform', bias_initializer='zeros', name=model_name+'_'+str(k), kernel_regularizer=None)(i_feat)
        i_feat = internal_layer
        if nnConfig['dropout_rate']:
            i_feat = Dropout(nnConfig['dropout_rate'], seed=20190128)(i_feat)

    layer_o = Dense(nnConfig['o_dim'], activation = nnConfig['activation_o'], kernel_initializer='glorot_uniform', bias_initializer='zeros', name=model_name+'_o')(i_feat)
    gDnnModel = keras.models.Model(inputs = input_vec, outputs = layer_o)

    return gDnnModel

def getBestModel(checkpoint_model_files):
    '''
    get the best DNN model from all saved checkpoint
    '''
    model_file_list = glob.glob(checkpoint_model_files + '-*-?.*.hdf5')
    best_model = ''
    best_score = np.inf
    for name in model_file_list:
        val_loss = float(name.split('-')[-1].split('.hdf5')[0])
        if val_loss < best_score:
            best_score = val_loss
            best_model = name
    print('INFO: Best model %s' % best_model)
    best_dnn_model = load_model(best_model)

    shutil.copyfile(best_model, checkpoint_model_files)
    #remove checkpoint files
    if settings.REMOVE_CHECKPOINT_MODEL:
        for name in model_file_list:
            os.remove(name)

    return best_dnn_model

def getBestModelFromHistory(checkpoint_model_files, dev_valid_value):
    '''
    get the best DNN model from all saved checkpoint
    '''
    best_epoch = np.argmin(dev_valid_value) + 1
    print(dev_valid_value)
    print(checkpoint_model_files + '-%02d-?.*.hdf5'%best_epoch)
    #model_file_list = glob.glob(checkpoint_model_files + '-%02d-?.*.hdf5'%best_epoch)
    model_file_list = glob.glob(checkpoint_model_files + '-%02d-*.*.hdf5'%best_epoch)
    print(model_file_list)
    if len(model_file_list) != 1:
        raise Exception('Warning: model name has same epoch')
    best_model = model_file_list[0]
    print('INFO: Best model %s' % best_model)
    best_dnn_model = load_model(best_model)

    shutil.copyfile(best_model, checkpoint_model_files)
    #remove checkpoint files
    if settings.REMOVE_CHECKPOINT_MODEL:
        #model_file_list = glob.glob(checkpoint_model_files + '-*-?.*.hdf5')
        model_file_list = glob.glob(checkpoint_model_files + '-*-*.*.hdf5')
        for name in model_file_list:
            os.remove(name)

    return best_dnn_model

def predictReport(x_insts, x_inst_index, model_set, model_type = None, topic_model_type = None, featureModelSet = None, x_insts_meta_embed = None):
    '''
    predict single input  instance
    '''
    if model_type == 'dnn':
        y_predict = np.zeros(shape = (x_insts.shape[0], settings.SEQ2INST_PARAM['predict_slot']), dtype='float32')
        gkey = x_inst_index
        if settings.DNN_CONFIG['model_type'] == 'embed':
            x_input = [x_insts, x_insts_meta_embed]
        else:
            x_input = x_insts
        y_predict = model_set[gkey].predict(x_input)

    if settings.SEQ2INST_PARAM['is_log_pv_y']:
        y_predict = np.floor(np.exp(y_predict) - settings.SEQ2INST_PARAM['value_avoid_overflow'] + 0.5)
    #
    y_predict[y_predict < 1.] = 0
    #
    return y_predict.astype('int32').tolist()


def scoreReport(y_truth, y_predict):
    if settings.SEQ2INST_PARAM['is_log_pv_y']:
        y_predict = np.floor(np.exp(y_predict) - settings.SEQ2INST_PARAM['value_avoid_overflow']
 + 0.5)
        y_truth = np.floor(np.exp(y_truth) - settings.SEQ2INST_PARAM['value_avoid_overflow']
 + 0.5)
    else:
        y_predict = np.floor(y_predict + 0.5)
    #
    y_predict[y_predict < 1.] = 0
    #
    metric_score = getMetric(y_truth, y_predict, is_display_detail=True)
    #print(metric_score)

    return metric_score, (y_truth, y_predict)


#test
if __name__ == '__main__':
    topic = '十年国开债中债估值'
    #i_train_data_file = '/home/gao/Work/aifintech/data/feat/美元兑人民币.pkl'
    #i_train_data_file = '/home/gao/Work/aifintech/data/feat/pmi.pkl'
    #i_train_data_file = '/home/gao/Work/aifintech/data/feat/cpi.pkl'
    #i_train_data_file = '/home/gao/Work/aifintech/data/feat/黄金.pkl'
    #i_train_data_file = '/home/gao/Work/aifintech/data/feat/十年国开债中债估值.pkl'
    #i_train_data_file = '/home/gao/Work/aifintech/data/feat/白银.pkl'
    #i_train_data_file = '/home/gao/Work/aifintech/data/feat/社会融资规模存量.pkl'
    i_train_data_file = '/home/gao/Work/aifintech/data/feat/{}.pkl'.format(topic)
    
    gmodel, feat_model, hs = fit(i_train_data_file,'','')

    with open(i_train_data_file, 'rb') as fl:
        data_set = pickle.load(fl)
    x_test, y_test = data_set['test'][:2]
    if is_feature_tuning:
        x_test = feat_transform(feat_model, x_test)

    #y_test = y_test[:, 0]
    #x_test = norm_me[1].transform(norm_me[0].transform(x_test))
    #x_test = norm_me[0].transform(x_test)
    y_pred = gmodel.predict(x_test)

    if settings.DNN_CONFIG['is_log']:
        y_pred = np.exp(y_pred)
    score_dev = mean_squared_error(y_pred, y_test)

    result_df = pd.DataFrame(columns = ['ds', 'actual', 'predict'])
    result_df['ds'] = data_set['test'][2]
    result_df['actual'], result_df['predict'] = y_test, y_pred

    result_file = '/home/gao/Work/deep_regression/exp/result/{}_dnn'.format(topic) + '_test.xlsx'
    result_df.to_excel(result_file)

    res = list(zip(y_pred, y_test))
    print(res[-4:])
    print('MSE: {}'.format(score_dev))

