#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This is for training/evaluating PV prediction 

Author: SHENG GAO
Date: Aug.21, 2018


'''

from __future__ import absolute_import

import sys, csv, os

sys.path.append('..')


import datetime
import random
import math
import cPickle as pickle
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics, linear_model, preprocessing
from sklearn import svm as SVM

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import settings
#from sklearn.preprocessing import Normalizer
from model.evalMetric import getMetric
from meradio.featureExtractor import computeFeature
from utils.clusterInst import clusterInstances


def fit(i_train_data_file, o_save_model_file, i_dev_data_file, model_method = 'retree', topic_type = None, is_hour_metric = True):
    '''
    automatic tuning model using dev data set
    Input:
        i_train_data_file: data file including training instances
        i_dev_data_file: data file including deve instances
        o_save_model_file: save best model to the file
        model_method: model method
        topic_type: method to cluster training instances. If None, using generic model
    '''

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
    if model_method == 'retree-m' or model_method == 'retree':
        #tune regression tree
        #if topic_type == '7-24':
        meModel, featureModel = tuneRegTree7a24(x_train, y_train, train_inst_index, x_dev, y_dev, dev_inst_index, model_method, topic_type, max_depth = 4)
    elif model_method == 'svr':
        meModel, featureModel = tuneSvr7a24(x_train, y_train, train_inst_index, x_dev, y_dev, dev_inst_index, model_method, topic_type)        
    else:
        raise Exception('model method error')

    #save model
    with open(o_save_model_file, 'wb') as o_fmodel:
        pickle.dump(meModel, o_fmodel)
        pickle.dump(featureModel, o_fmodel)
        #pickle.dump(xtransform, o_fmodel)
    '''
    try:
        pyarrow.serialize_to([meModel, featureModel], o_save_model_file)
    except:
        raise Exception('Save model to %s error' % o_save_model_file)
    '''

    metric_score, _ = scoreReport(x_train, y_train, train_inst_index, meModel, model_method, topic_type, is_hour_metric, feature_model_set = featureModel)

    print('Average performance in train: RMSE %f, SMAPE: %f, R2: %f' % (metric_score['rmse'], metric_score['smape'], metric_score['r2']))


def scoreReport(X_feat, Y_target, inst_index, meModel, method, topic_type, is_hour_metric, feature_model_set = None):
    #get performance in train set
    if isinstance(meModel, list):
        if feature_model_set:
            X_feat, _ = transformFeature(feature_model_set['generic'], X_feat)

        #it is slot based model         
        y_predict = np.zeros(np.array(Y_target).shape)
        for t in xrange(settings.SEQ2INST_PARAM['predict_slot']):                
            y_predict[:,t] = meModel[t].predict(X_feat)
        #transform back raw value            
        if feature_model_set:
            if settings.FEATURE_PREPROCESS_CONFIG['isGaussScaleY']:
                y_predict = feature_model_set['generic']['y'].inverse_transform(y_predict)                
    elif isinstance(meModel, dict):
        #it is slot based model
        y_predict = np.zeros(np.array(Y_target).shape)
        for gkey, gidx in inst_index.iteritems():
            if gidx is None or len(gidx) < 1:
                continue
            if topic_type:
                if gkey == 'generic':
                    continue
            X_feat2 = X_feat[gidx, :]

            if feature_model_set:
                X_feat2, _ = transformFeature(feature_model_set[gkey], X_feat2)

            #if transform_func:
            #    pass #X_feat2 = transform_func[gkey].transform(X_feat2)

            if len(meModel[gkey]) == 1:
                y_predict[gidx, :] = meModel[gkey][0].predict(X_feat2)
                #transform back raw value            
                if feature_model_set:
                    if settings.FEATURE_PREPROCESS_CONFIG['isGaussScaleY']:
                        y_predict[gidx, :] = feature_model_set[gkey]['y'].inverse_transform(y_predict[gidx, :])
                    #Y_target[gidx, :] = feature_model_set[gkey]['y'].inverse_transform(Y_target[gidx, :])

            else:
                for t in xrange(settings.SEQ2INST_PARAM['predict_slot']):                
                    y_predict[gidx, t] = meModel[gkey][t].predict(X_feat2)
                #transform back raw value            
                if feature_model_set:
                    if settings.FEATURE_PREPROCESS_CONFIG['isGaussScaleY']:
                        y_predict[gidx, :] = feature_model_set[gkey]['y'].inverse_transform(y_predict[gidx, :])
                        #Y_target[gidx, :] = feature_model_set[gkey]['y'].inverse_transform(Y_target[gidx, :])
    else:
        y_predict = meModel.predict(X_feat)
        #transform back raw value            
        if feature_model_set:
            gkey = 'generic'
            if settings.FEATURE_PREPROCESS_CONFIG['isGaussScaleY']:
                y_predict = feature_model_set[gkey]['y'].inverse_transform(y_predict)
            #Y_target = feature_model_set[gkey]['y'].inverse_transform(Y_target)

    #
    #print(y_predict.shape)
    if settings.SEQ2INST_PARAM['is_log_pv_y']:
        y_predict = np.floor(np.exp(y_predict) - settings.SEQ2INST_PARAM['value_avoid_overflow']
 + 0.5)
        y_truth = np.floor(np.exp(Y_target) - settings.SEQ2INST_PARAM['value_avoid_overflow']
 + 0.5)
    else:
        y_predict = np.floor(y_predict + 0.5)
        y_truth = Y_target
    #
    y_predict[y_predict < 1.] = 0
    #
    metric_score = getMetric(y_truth, y_predict, is_display_detail=True)

    #sc = metric_score['r2']
    #print('R2 in train %f ' % sc)
    #sc = metric_score['smape']
    #print('mean squared error in train %f ' % sc)
    return metric_score, (X_feat, y_truth, y_predict)

def predictReport(X_feat, inst_index, meModel, method, topic_type, featureModelSet):    
    #predict based on input
    X_feat2 = X_feat

    if featureModelSet != None and featureModelSet != {}:
        X_feat2, _ = transformFeature(featureModelSet[inst_index], x_feats = np.array(X_feat2))
        X_feat2 = X_feat2.tolist()

    if isinstance(meModel, list):
        #it is slot based model         
        y_predict = np.zeros(np.array(Y_target).shape)
        for t in xrange(settings.SEQ2INST_PARAM['predict_slot']):                
            y_predict[:,t] = meModel[t].predict(X_feat2)
    elif isinstance(meModel, dict):
        #it is slot based model
        y_predict = np.zeros(np.array((1, settings.SEQ2INST_PARAM['predict_slot'])))
        if len(meModel[inst_index]) == 1:
            y_predict = meModel[inst_index][0].predict(X_feat2)
        else:
            for t in xrange(settings.SEQ2INST_PARAM['predict_slot']):                
                y_predict[:, t] = meModel[inst_index][t].predict(X_feat2)

    else:
        y_predict = meModel.predict(X_feat2)

    if featureModelSet != None and featureModelSet != {}:
        if settings.FEATURE_PREPROCESS_CONFIG['isGaussScaleY']:
            y_predict = featureModelSet[inst_index]['y'].inverse_transform(y_predict)

    #print(y_predict.shape)
    if settings.SEQ2INST_PARAM['is_log_pv_y']:
        y_predict = np.floor(np.exp(y_predict) - settings.SEQ2INST_PARAM['value_avoid_overflow']
 + 0.5)
    else:
        y_predict = np.floor(y_predict + 0.5)
    #
    y_predict[y_predict < 1.] = 0
    #
    return y_predict


def tuneRegTree7a24(train_insts_x, train_insts_y, train_inst_cluster_index, dev_insts_x, dev_insts_y, dev_inst_cluster_index, model_method, topic_type, max_depth = 5):
    '''
    tune model based training set and dev set

    Input:
        train_insts_x, train_insts_y, train_inst_cluster_index: training instances, X, Y, topic_index
        dev_insts_x, dev_insts_y, dev_inst_cluster_index: dev instances, X, Y, topic_index
        model_method: model type
        topic_type: method used to cluster instances, e.g 7-24
        max_depth: maximal depth of tree to be searched

    return best model
    '''
    criterion_name = 'mse' # 'mae' #
    k_depth_range = [None] + range(1, max_depth + 1) #

    #first train generic model using all data
    bestModelSet = dict()
    featureModelSet = {}
    bestModel = None
    model_select_metric = settings.MODEL_SELECTION_METRIC
    if model_select_metric == 'r2':
        bestScore = -np.inf
    else:
        bestScore = np.inf

    gkey = 'generic'
    x_feats = train_insts_x[train_inst_cluster_index[gkey], :]
    y_target = train_insts_y[train_inst_cluster_index[gkey], :]

    #feature normalize
    feature_model = fitFeature(x_feats, y_target)
    if feature_model:
        x_feats, y_target = transformFeature(feature_model, x_feats, y_target)
        dev_insts_x, dev_insts_y = transformFeature(feature_model, dev_insts_x, dev_insts_y)
        featureModelSet[gkey] = feature_model
    #
    for k_depth in k_depth_range:
        my_param = {'depth': k_depth, 'criterion': criterion_name }
        cur_model = fitSpecific(x_feats, y_target, my_param, model_method)
        score2 = scoreModel(dev_insts_x, dev_insts_y, cur_model, feature_model, is_hour_metric=True, is_display_detail=False)

        if model_select_metric == 'r2':
            score = score2['r2']
            if score > bestScore:
                bestModel = cur_model
                bestScore = score
        else:
            score = score2[model_select_metric]
            if score < bestScore:
                bestModel = cur_model
                bestScore = score

    bestModelSet[gkey] = bestModel

    #xtranform = {}
    #xtranform['generic'] = Normalizer(norm = 'l2')
    #xtranform['generic'].fit(train_insts_x)
    #train_insts_x = xtranform['generic'].transform(train_insts_x)

    #end of generic model training

    #start to train topic model, optimzied for each topic based instances
    if topic_type == '7-24' or topic_type == '7-24-inst'or topic_type == '7-inst'or topic_type == '24-inst':
        model_counter = 0
        for gkey, gid in train_inst_cluster_index.iteritems():
            #for each topic, find the best model to predict 72-hour
            print('Start to train model %d %s' % (model_counter, gkey))
            gid_dev = dev_inst_cluster_index[gkey]

            if gid is None or len(gid) < 1 or gid_dev is None or len(gid_dev) < 1:
                #if topic based training instance is few, use generic model
                bestModelSet[gkey] = bestModelSet['generic']
                if feature_model:
                    featureModelSet[gkey] = featureModelSet['generic']
                #xtranform[gkey] = xtranform['generic']
                model_counter += 1
                continue
            if gkey == 'generic':
                continue
            x_feats = train_insts_x[gid, :]
            y_target = train_insts_y[gid, :]            
            x_feats_dev = dev_insts_x[gid_dev, :]
            y_target_dev = dev_insts_y[gid_dev, :]

            #feature normalize
            feature_model = fitFeature(x_feats, y_target)
            if feature_model:
                x_feats, y_target = transformFeature(feature_model, x_feats, y_target)
                x_feats_dev, y_target_dev = transformFeature(feature_model, x_feats_dev, y_target_dev)
                featureModelSet[gkey] = feature_model

            #tranform feature
            #xtranform[gkey] = preprocessing.Normalizer(norm = 'l2')
            #xtranform[gkey].fit(x_feats)
            #x_feats = xtranform[gkey].transform(x_feats)
            #x_feats_dev = xtranform[gkey].transform(x_feats_dev)
            #
            bestModel = None
            if model_select_metric == 'r2':
                bestScore = -np.inf
            else:
                bestScore = np.inf
            #get instances for the cluster
            for k_depth in k_depth_range:
                my_param = {'depth': k_depth, 'criterion': criterion_name }
                cur_model = fitSpecific(x_feats, y_target, my_param, model_method)
                score2 = scoreModel(x_feats_dev, y_target_dev, cur_model, feature_model, is_hour_metric=True, is_display_detail=False)
                
                if model_select_metric == 'r2':
                    score = score2['r2']
                    if score > bestScore:
                        bestModel = cur_model
                        bestScore = score
                else:
                    score = score2[model_select_metric]
                    if score < bestScore:
                        bestModel = cur_model
                        bestScore = score

            bestModelSet[gkey] = bestModel
    return bestModelSet, featureModelSet #, xtranform

def fitSpecific(x_feats, y_target, my_param, model_method, o_save_model_file = None):
    if settings.IS_CONFIG_TREE:
        min_insts_split = max(min(int(float(x_feats.shape[0]) * settings.TREE_CONFIG['split_ratio']), 20), 2)
        min_insts_leaf = max(min(int(float(x_feats.shape[0]) * settings.TREE_CONFIG['leaf_ratio']), 10), 1)
    else:        
        min_insts_split = 2
        min_insts_leaf = 1

    cur_model = []
    if model_method == 'retree':
        k_depth, criterion_name = my_param['depth'], my_param['criterion']
        #train model for the specific model parameters
        for t in xrange(settings.SEQ2INST_PARAM['predict_slot']):                
            slotModel = DecisionTreeRegressor(max_depth = k_depth, criterion = criterion_name, random_state=20180822, min_samples_split = min_insts_split, min_samples_leaf = min_insts_leaf)
            slotModel.fit(x_feats, y_target[:, t])
            cur_model.append(slotModel)
    elif model_method == 'retree-m':
        k_depth, criterion_name = my_param['depth'], my_param['criterion']
        #regression model for optimize multiple output
        slotModel = DecisionTreeRegressor(max_depth = k_depth, criterion = criterion_name, random_state=20180822, min_samples_split = min_insts_split, min_samples_leaf = min_insts_leaf) #, min_samples_split = 6, min_samples_leaf = 3)
        slotModel.fit(x_feats, y_target)
        cur_model.append(slotModel)
    elif model_method == 'svr':
        for t in xrange(settings.SEQ2INST_PARAM['predict_slot']): 
            sys.stdout.write('%d ' % t)
            sys.stdout.flush()

            slotModel = SVM.LinearSVR() #SVR() #
            '''\
                C = my_param['cost'], \
                loss = my_param['loss'], \
                epsilon = my_param['epsilon'], \
                dual = my_param['dual'],\
                fit_intercept = my_param['fit_intercept'],\
                max_iter = my_param['max_iter'],
                random_state = 20180822
                )
            '''
            slotModel.fit(x_feats, y_target[:,t])
            cur_model.append(slotModel)
    else:
        raise Exception('model method only support regtree, regtree-m')

    return cur_model

def fitFeature(x_feats, y_target):
    #do feature preprocessing, Gaussian-normalize
    feature_model = None
    if settings.FEATURE_PREPROCESS_CONFIG['isGaussScale']:
        feature_model = {'x': preprocessing.StandardScaler().fit(x_feats)}
        #x_feats = feature_model['x'].transform(x_feats) 
        feature_model['y'] = preprocessing.StandardScaler().fit(y_target)
        #y_target = feature_model['y'].transform(y_target) 
    else:
        pass
    return feature_model
    #
def transformFeature(feature_model, x_feats = None, y_target = None):
    #do feature preprocessing, Gaussian-normalize
    if isinstance(x_feats, np.ndarray):
        if settings.FEATURE_PREPROCESS_CONFIG['isGaussScaleX']:
            x_feats = feature_model['x'].transform(x_feats) 
    if isinstance(y_target, np.ndarray):
        if settings.FEATURE_PREPROCESS_CONFIG['isGaussScaleY']:
            y_target = feature_model['y'].transform(y_target) 
    return x_feats, y_target

def tuneSvr7a24(train_insts_x, train_insts_y, train_inst_cluster_index, dev_insts_x, dev_insts_y, dev_inst_cluster_index, model_method, topic_type):
    '''
    tune support vector regression based training set and dev set

    Input:
        train_insts_x, train_insts_y, train_inst_cluster_index: training instances, X, Y, topic_index
        dev_insts_x, dev_insts_y, dev_inst_cluster_index: dev instances, X, Y, topic_index
        model_method: model type
        topic_type: method used to cluster instances, e.g 7-24
        max_depth: maximal depth of tree to be searched

    return best model
    '''
    my_param = {}
    my_param['loss'] = 'epsilon_insensitive' # squared_epsilon_insensitive
    my_param['cost'] = 1.0
    my_param['fit_intercept'] = True
    my_param['epsilon'] = 0.1
    my_param['random_state'] = 20180904
    my_param['max_iter'] = 1000
    my_param['dual'] = False

    #first train generic model using all data
    bestModelSet = dict()
    featureModelSet = {}
    bestModel = None
    model_select_metric = settings.MODEL_SELECTION_METRIC
    if model_select_metric == 'r2':
        bestScore = -np.inf
    else:
        bestScore = np.inf

    gkey = 'generic'
    x_feats = train_insts_x[train_inst_cluster_index[gkey], :]
    y_target = train_insts_y[train_inst_cluster_index[gkey], :]

    #feature normalize
    feature_model = fitFeature(x_feats, y_target)
    if feature_model:
        x_feats, y_target = transformFeature(feature_model, x_feats, y_target)
        dev_insts_x, dev_insts_y = transformFeature(feature_model, dev_insts_x, dev_insts_y)
        featureModelSet[gkey] = feature_model
    #

    if True: #for k_depth in k_depth_range:
        cur_model = fitSpecific(x_feats, y_target, my_param, model_method)
        score2 = scoreModel(dev_insts_x, dev_insts_y, cur_model, feature_model, is_hour_metric=True, is_display_detail=False)

        if model_select_metric == 'r2':
            score = score2['r2']
            if score > bestScore:
                bestModel = cur_model
                bestScore = score
        else:
            score = score2[model_select_metric]
            if score < bestScore:
                bestModel = cur_model
                bestScore = score

    bestModelSet[gkey] = bestModel

    #xtranform = {}
    #xtranform['generic'] = Normalizer(norm = 'l2')
    #xtranform['generic'].fit(train_insts_x)
    #train_insts_x = xtranform['generic'].transform(train_insts_x)

    #end of generic model training

    #start to train topic model, optimzied for each topic based instances
    '''
    if topic_type == '7-24' or topic_type == '7-24-inst':
        model_counter = 0
        for gkey, gid in train_inst_cluster_index.iteritems():
            #for each topic, find the best model to predict 72-hour
            print('Start to train model %d %s' % (model_counter, gkey))
            gid_dev = dev_inst_cluster_index[gkey]

            if gid is None or len(gid) < 1 or gid_dev is None or len(gid_dev) < 1:
                #if topic based training instance is few, use generic model
                bestModelSet[gkey] = bestModelSet['generic']
                #xtranform[gkey] = xtranform['generic']
                model_counter += 1
                continue
            if gkey == 'generic':
                continue
            x_feats = train_insts_x[gid, :]
            y_target = train_insts_y[gid, :]            
            x_feats_dev = dev_insts_x[gid_dev, :]
            y_target_dev = dev_insts_y[gid_dev, :]
            #tranform feature
            #xtranform[gkey] = preprocessing.Normalizer(norm = 'l2')
            #xtranform[gkey].fit(x_feats)
            #x_feats = xtranform[gkey].transform(x_feats)
            #x_feats_dev = xtranform[gkey].transform(x_feats_dev)
            #
            bestModel = None
            if model_select_metric == 'r2':
                bestScore = -np.inf
            else:
                bestScore = np.inf
            #get instances for the cluster
            for k_depth in k_depth_range:
                cur_model = fitSpecific(x_feats, y_target, k_depth, criterion_name, model_method)
                score2 = scoreModel(x_feats_dev, y_target_dev, cur_model, is_hour_metric=True, is_display_detail=False)
                
                if model_select_metric == 'r2':
                    score = score2['r2']
                    if score > bestScore:
                        bestModel = cur_model
                        bestScore = score
                else:
                    score = score2[model_select_metric]
                    if score < bestScore:
                        bestModel = cur_model
                        bestScore = score

            bestModelSet[gkey] = bestModel
    '''
    return bestModelSet, featureModelSet #, xtranform

def scoreModel(X_feat, Y_target, cur_model, feature_model, is_hour_metric = False, is_display_detail = False, model_method = None, o_save_model_file = None):
    '''
    evaluate model on input feture set.
    '''
    #get performance metric
    print('Total test instance %d' % len(X_feat))
    pred_result = np.zeros(np.array(Y_target).shape)
    if len(cur_model) == settings.SEQ2INST_PARAM['predict_slot']:
        for t in xrange(settings.SEQ2INST_PARAM['predict_slot']):                
            pred_result[:,t] = cur_model[t].predict(X_feat)
    elif len(cur_model) == 1:
        pred_result = cur_model[0].predict(X_feat)
    else:
        raise Exception('Warning: model trained error')

    #transform back raw value            
    if feature_model != {} and feature_model != None:
        if settings.FEATURE_PREPROCESS_CONFIG['isGaussScaleY']:
            pred_result = feature_model['y'].inverse_transform(pred_result)
            Y_target = feature_model['y'].inverse_transform(Y_target)

    if settings.SEQ2INST_PARAM['is_log_pv_y']:
        y_predict = np.floor(np.exp(pred_result) - settings.SEQ2INST_PARAM['value_avoid_overflow']
 + 0.5)
        y_truth = np.floor(np.exp(Y_target) - settings.SEQ2INST_PARAM['value_avoid_overflow']
 + 0.5)
    else:
        y_predict = pred_result
        y_truth = Y_target
    #
    y_predict[y_predict < 1.] = 0
    #

    metric_score = getMetric(y_truth, y_predict, is_display_detail = is_display_detail)

    return metric_score


def vaildateTrainInsts(i_insts_x, i_insts_y, i_insts_meta):
    '''
    function to analyze duplicate X instances in training,
    '''
    key_index_insts = {}
    for i_id, (x, y) in enumerate(zip(i_insts_x, i_insts_y)):
        #build key
        ky = buildVectKey(x)
        if ky in key_index_insts:
            key_index_insts[ky].append(i_id)
        else:
            key_index_insts[ky] = [i_id]
    #count how many insts where x is same but y is different
    o_insts_x, o_insts_y, o_insts_meta = [], [], []
    dup_inst_x_count = 0
    dup_inst_xy_count = 0
    for ky, vid in key_index_insts.iteritems():
        if len(vid) > 1:
            pre_v0 = i_insts_y[vid[0]]
            same_y_count = 1
            for v0 in vid[1:]:
                if i_insts_y[v0] == pre_v0:
                    same_y_count += 1
            if same_y_count == len(vid):
                dup_inst_xy_count += len(vid)
                o_insts_x.append(i_insts_x[vid[0]])
                o_insts_y.append(i_insts_y[vid[0]])
                o_insts_meta.append(i_insts_meta[vid[0]])
            else:
                dup_inst_x_count += len(vid)
        else:
            o_insts_x.append(i_insts_x[vid[0]])
            o_insts_y.append(i_insts_y[vid[0]])
            o_insts_meta.append(i_insts_meta[vid[0]])

    print('%d toal instances' % len(i_insts_x))
    print('%d instances with same x but different y' % dup_inst_x_count)
    print('%d instances with same x and y' % dup_inst_xy_count)
    print('%d after remove above cases' % len(o_insts_x))
    return o_insts_x, o_insts_y, o_insts_meta

def buildVectKey(x):
    return ','.join([str(k) for k in x])


def TEST_vaildateTrainInsts():
    #load training instance
    i_train_data_file = '/home/gao/work/shared/EXP/temp/network/train_inst.pkl'
    '''
    with open(i_train_data_file, 'rb') as i_fdata:
        train_insts_x_meta = pickle.load(i_fdata)
        train_insts_x = pickle.load(i_fdata)
        train_insts_y = pickle.load(i_fdata)

    train_insts_x, train_insts_y, train_insts_x_meta = vaildateTrainInsts(train_insts_x, train_insts_y, train_insts_x_meta)
    '''

if __name__ == '__main__':
    MODEL_TYPE = 'retree-m' #'lasso' #lasso-m' # #'retree' #lasso'

    test_pipeline = [TEST_vaildateTrainInsts] #TEST_train, TEST_eval]#, ]#]#, ]#
    for f in test_pipeline:
        f()
    
