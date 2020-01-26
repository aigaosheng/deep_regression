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
import cPickle as pickle
#from pyarrow import deserialize_from
import numpy as np
import tensorflow as tf
from keras import backend as K, regularizers
import json
import glob, shutil


import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Input, Embedding, LSTM, BatchNormalization, Dropout
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint


#from sklearn.preprocessing import Normalizer
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.evalMetric import getMetric
from meradio.featureExtractor import computeFeature
from utils.clusterInst import clusterInstances
import settings
from model.dnnMetric import METRIC_LIST

def fit(i_train_data_file, o_save_model_file, i_dev_data_file, model_method = 'dnn', topic_type = None, is_hour_metric = True):
    '''
    automatic tuning model using dev data set
    Input:
        i_train_data_file: data file including training instances
        i_dev_data_file: data file including deve instances
        o_save_model_file: save best model to the file
        model_method: model method
        topic_type: method to cluster training instances. If None, using generic model
    '''

    #load training instance
    '''try:
        train_insts_x, train_insts_x_meta, train_insts_y = loadFeatureInstsMemmap(i_train_data_file)
    except:
        raise Exception('File %s not found' % i_train_data_file)

    #convert raw pv/meta to raw feature vector
    featureMemmap_file_folder = os.path.join(os.path.dirname(i_train_data_file), 'featuremap')
    if not os.path.exists(featureMemmap_file_folder):
        os.mkdir(featureMemmap_file_folder)
    featureMemmap_file = os.path.join(os.path.dirname(i_train_data_file), 'featuremap', os.path.basename(i_train_data_file))'''
    x_train_meta_embed, x_train, y_train, x_train_meta, Weight_feature = computeFeature(i_train_data_file, settings.IS_REMOVE_LOW_PV_VV, sampling_rate = settings.REDUCE_SAMPLE_SIZE_RATIO) 

    print('INFO: %d instances loaded from from %s' % (x_train.shape[0], i_train_data_file))
    #Clustering instances into topic
    print('Clustering dev instances using %s' % topic_type)
    print('Clustering training instances using %s' % topic_type)
    train_inst_index = clusterInstances(x_train_meta, x_train, y_train, method = topic_type)
    '''
    try:
        dev_insts_x, dev_insts_x_meta, dev_insts_y = loadFeatureInstsMemmap(i_dev_data_file)
    except:
        raise Exception('File %s not found' % i_dev_data_file)

    #convert raw pv/meta to raw feature vector
    featureMemmap_file_folder = os.path.join(os.path.dirname(i_dev_data_file), 'featuremap')
    if not os.path.exists(featureMemmap_file_folder):
        os.mkdir(featureMemmap_file_folder)
    featureMemmap_file = os.path.join(os.path.dirname(i_dev_data_file), 'featuremap', os.path.basename(i_dev_data_file))'''
    if i_dev_data_file != None and i_dev_data_file != '':
        try:
            x_dev_meta_embed, x_dev, y_dev, x_dev_meta, _ = computeFeature(i_dev_data_file, settings.IS_REMOVE_LOW_PV_VV_TEST) #, dev_insts_x, insts_x_meta = dev_insts_x_meta, insts_y = dev_insts_y) 
            dev_inst_index =  clusterInstances(x_dev_meta, x_dev, y_dev, method = topic_type)    
            print('INFO: %d instances loaded from from %s' % (x_dev.shape[0], i_dev_data_file))
        except:
            x_dev_meta_embed, x_dev, y_dev, x_dev_meta, dev_inst_index = None, None, None,None, None
    else:
        x_dev_meta_embed, x_dev, y_dev, x_dev_meta, dev_inst_index = None, None, None,None, None

    #train model
    if model_method == 'dnn':
        meModel = dnnLearner(Weight_feature, x_train, y_train, train_inst_index, x_dev, y_dev, dev_inst_index, topic_type, o_save_model_file, x_train_meta_embed = x_train_meta_embed, x_dev_meta_embed = x_dev_meta_embed)
    else:
        raise Exception('model method error')

    #save model
    #print(meModel)
    #for gkey, mm in meModel.iteritems():
    #    mm.save(o_save_model_file + '.' + gkey)

    '''
    try:
        pyarrow.serialize_to([meModel, featureModel], o_save_model_file)
    except:
        raise Exception('Save model to %s error' % o_save_model_file)
    '''

    #metric_score, _ = scoreReport(train_insts_x, train_insts_y, train_inst_index, meModel, model_method, topic_type, is_hour_metric, feature_model_set = featureModel)

    #print('Average performance in train: RMSE %f, SMAPE: %f, R2: %f' % (metric_score['rmse'], metric_score['smape'], metric_score['r2']))




def dnnLearner(train_inst_weight, train_insts_x, train_insts_y, train_inst_cluster_index, dev_insts_x, dev_insts_y, dev_inst_cluster_index, topic_type = None, o_save_model=None, x_train_meta_embed= None, x_dev_meta_embed= None):
    '''
    train model based training set and dev set

    Input:
        train_insts_x, train_insts_y, train_inst_cluster_index: training instances, X, Y, topic_index
        dev_insts_x, dev_insts_y, dev_inst_cluster_index: dev instances, X, Y, topic_index
        model_method: model type

    return best model
    '''
    if o_save_model:
        o_save_model_file_name = o_save_model + '.' + 'generic'
    else:
        raise Exception('Warning: must set save model file names')

    bestModelSet = {}
    #bestModelSet['generic'] = fitClusterModel(train_inst_weight, train_insts_x, train_insts_y, dev_x = dev_insts_x, dev_y = dev_insts_y, model_name = 'generic', save_model = o_save_model_file_name)
    _, valid_value = fitClusterModel(train_inst_weight, train_insts_x, train_insts_y, dev_x = dev_insts_x, dev_y = dev_insts_y, model_name = 'generic', save_model = o_save_model_file_name, x_train_meta_embed = x_train_meta_embed, x_dev_meta_embed = x_dev_meta_embed)
    #bestModelSet['generic'] = getBestModel(o_save_model_file_name)
    bestModelSet['generic'] = getBestModelFromHistory(o_save_model_file_name, valid_value)
    #bestModelSet['generic'].save(o_save_model + '.' + 'generic')


    #return bestModelSet
    #mykey = 'd7h'

    if topic_type == '7-24' or topic_type == '7-24-inst' or topic_type == '24-inst' or topic_type == '7-inst':
        model_counter = 0
        for gkey, gid in train_inst_cluster_index.iteritems():
            #for each topic, find the best model to predict 72-hour
            #if gkey != mykey:
            #    continue
            
            print('Start to train model %d %s' % (model_counter, gkey))
            
            if gid is None or len(gid) < 1:
                #if topic based training instance is few, use generic model
                #bestModelSet[gkey] = None
                model_counter += 1
                continue
            if gkey == 'generic':
                continue
            o_save_model_file_name = o_save_model + '.' + gkey

            x_feats = train_insts_x[gid, :]
            y_target = train_insts_y[gid, :]
            weight_feats = train_inst_weight[gid]
            
            if dev_insts_x is not None:
                dev_gid = dev_inst_cluster_index[gkey]
                #bestModelSet[gkey] = fitClusterModel(weight_feats, x_feats, y_target, dev_x = dev_insts_x[dev_gid, :], dev_y = dev_insts_y[dev_gid, :], model_name=gkey, save_model = o_save_model_file_name)
                _, valid_value = fitClusterModel(weight_feats, x_feats, y_target, dev_x = dev_insts_x[dev_gid, :], dev_y = dev_insts_y[dev_gid, :], model_name=gkey, save_model = o_save_model_file_name, x_train_meta_embed = x_train_meta_embed, x_dev_meta_embed = x_dev_meta_embed)
                #bestModelSet[gkey] = getBestModel(o_save_model_file_name)
            else:
                #bestModelSet[gkey] = fitClusterModel(weight_feats, x_feats, y_target, model_name=gkey, save_model = o_save_model_file_name)
                _, valid_vaule = fitClusterModel(weight_feats, x_feats, y_target, model_name=gkey, save_model = o_save_model_file_name, x_train_meta_embed = x_train_meta_embed, x_dev_meta_embed = x_dev_meta_embed)
                #bestModelSet[gkey] = getBestModel(o_save_model_file_name)
            bestModelSet[gkey] = getBestModelFromHistory(o_save_model_file_name, valid_value)
            
            model_counter += 1
            #bestModelSet[gkey].save(o_save_model + '.' + gkey)
    
    if dev_insts_x is not None:
        pass
        #metric_score_dev, y_true_predict_pair = modelEval(bestModelSet, dev_inst_cluster_index, dev_insts_x, dev_insts_y)

    return bestModelSet

def fitClusterModel(weight_inst, insts_x, insts_y, dev_x = None, dev_y = None, model_name = 'gs', save_model = './gsmodel.pkl',x_train_meta_embed = None, x_dev_meta_embed = None):
    '''
    fit DNN model using input data
    if gDnnModel is not None: start training from gDnnModel, which is previously trained model
    '''
    nnConfig = settings.DNN_CONFIG
    nnConfig['i_dim'] = insts_x.shape[1]

    tf.set_random_seed(1234)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    #
    if nnConfig['model_type'].lower() == 'embed':
        gDnnModel = defineDnnModelEmbed(nnConfig, model_name)
    else:
        gDnnModel = defineDnnModel(nnConfig, model_name)

    sgd = SGD(lr = nnConfig['learning_rate'], decay = 1e-6, momentum = 0.9, nesterov=True)

    if nnConfig['val_metric'] not in METRIC_LIST:
        raise ValueError('Warning: val_metric must be %s' % ','.join(METRIC_LIST.keys()))
    #keras.metrics.smapeMetric = nnConfig['val_metric']
    '''    
    if not hasattr(keras.metrics, nnConfig['val_metric']):
        setattr(keras.metrics, nnConfig['val_metric'], METRIC_LIST[nnConfig['val_metric']])
    if not hasattr(keras.losses, nnConfig['loss']):
        setattr(keras.losses, nnConfig['loss'], METRIC_LIST[nnConfig['loss']])
    '''
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
    callbacks_list = [checkpoint]

    # Fit the model
    if nnConfig['model_type'].lower() == 'embed':
        input_list_train = [insts_x, x_train_meta_embed]
        input_list_dev= [dev_x, x_dev_meta_embed]
    else:
        input_list_train = insts_x
        input_list_dev= dev_x

    if settings.IS_SAMPLE_WEIGHT_USED:
        if dev_x is not None:
            history = gDnnModel.fit(input_list_train, insts_y, epochs = nnConfig['n_epoch'], batch_size = nnConfig['n_batch_size'], sample_weight = weight_inst, validation_data = (input_list_dev, dev_y), callbacks = callbacks_list)
        else:
            history = gDnnModel.fit(input_list_train, insts_y, epochs = nnConfig['n_epoch'], batch_size = nnConfig['n_batch_size'], sample_weight = weight_inst, validation_split = nnConfig['validation_split'], callbacks = callbacks_list)
    else:
        if dev_x is not None:
            history = gDnnModel.fit(input_list_train, insts_y, epochs = nnConfig['n_epoch'], batch_size = nnConfig['n_batch_size'], validation_data = (input_list_dev, dev_y), callbacks = callbacks_list)
        else:
            history = gDnnModel.fit(input_list_train, insts_y, epochs = nnConfig['n_epoch'], batch_size = nnConfig['n_batch_size'], validation_split = nnConfig['validation_split'], callbacks = callbacks_list)
    #print(history.history)

    #save training log to file
    '''tempfile_name = 'train_' + str(random.randint(0,100)) + '.log'
    with open(tempfile_name, 'wt') as oTempLog:
        json.dump(history.history, oTempLog, indent=2)
    '''
    return gDnnModel, history.history['val_' + nnConfig['val_metric']]


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


def defineDnnModelEmbed(nnConfig, model_name):
    '''
    Embed meta data using DNN model
    '''
    input_vec = Input(shape = (nnConfig['i_dim'],), name=model_name+'_i')
    i_feat = input_vec
    for k, layer_size in enumerate(nnConfig['layer_stack']):
        if nnConfig['is_batch_norm']:
            i_feat = BatchNormalization()(i_feat)
        internal_layer = Dense(layer_size, activation = nnConfig['activation_i'], kernel_initializer='glorot_uniform', bias_initializer='zeros', name=model_name+'_'+str(k))(i_feat)
        i_feat = internal_layer
        if nnConfig['dropout_rate']:
            i_feat = Dropout(nnConfig['dropout_rate'], seed=20190128)(i_feat)

    #embed meta
    input_vec_meta = Input(shape = (nnConfig['seq_length'],), dtype='int32', name=model_name+'_i_meta')
    input_vec_meta_embed = Embedding(input_dim = settings.EMBED_DIM_MAX, output_dim = nnConfig['embed_dim'], input_length = nnConfig['seq_length'])(input_vec_meta)
    o_rnn = LSTM(nnConfig['n_lstm_dim'])(input_vec_meta_embed)    
    
    #
    if settings.SEQ2INST_PARAM['is_log_pv_y']:
        layer_atten = Dense(nnConfig['meta_layer_dim'], activation = 'tanh', kernel_initializer='glorot_uniform', bias_initializer='zeros', name=model_name+'_atten_1')(o_rnn)
        layer_o_fw = keras.layers.add([i_feat, layer_atten])
    else:
        layer_atten = Dense(nnConfig['meta_layer_dim'], activation = 'softmax', kernel_initializer='glorot_uniform', bias_initializer='zeros', name=model_name+'_atten_1')(o_rnn)
        layer_o_fw = keras.layers.multiply([i_feat, layer_atten])
    #layer_o_fw = keras.layers.add([internal_layer, K.log(layer_atten)])
    layer_o = Dense(nnConfig['o_dim'], activation = nnConfig['activation_o'], kernel_initializer='glorot_uniform', bias_initializer='zeros', name=model_name+'_o')(layer_o_fw)
    #
    gDnnModel = keras.models.Model(inputs = [input_vec, input_vec_meta], outputs = layer_o)

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
    model_file_list = glob.glob(checkpoint_model_files + '-%02d-?.*.hdf5'%best_epoch)
    print(model_file_list)
    if len(model_file_list) != 1:
        raise Exception('Warning: model name has same epoch')
    best_model = model_file_list[0]
    print('INFO: Best model %s' % best_model)
    best_dnn_model = load_model(best_model)

    shutil.copyfile(best_model, checkpoint_model_files)
    #remove checkpoint files
    if settings.REMOVE_CHECKPOINT_MODEL:
        model_file_list = glob.glob(checkpoint_model_files + '-*-?.*.hdf5')
        for name in model_file_list:
            os.remove(name)

    return best_dnn_model

def modelEval(modelSet, eval_insts_index, eval_insts_x, eval_insts_y, y_predict = None, x_test_meta_embed = None):
    '''
    test model
    '''
    if not isinstance(y_predict, np.ndarray) and y_predict == None:
        y_predict = np.zeros(eval_insts_y.shape)
    for gkey, gid in eval_insts_index.iteritems():
        x_feats = eval_insts_x[gid, :]
        if gkey not in modelSet:
            gkey = 'generic'
        
        if settings.DNN_CONFIG['model_type'].lower() == 'embed':
            input_list_test = [x_feats, x_test_meta_embed[gid, :]]
        else:
            input_list_test = x_feats

        y_predict[gid, :] = modelSet[gkey].predict(input_list_test)
    
    metric_score, y_p_pair = scoreReport(eval_insts_y, y_predict)      
    return metric_score, y_p_pair

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

