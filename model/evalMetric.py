#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This is to define the performance metric function to report prediction score

Author: SHENG GAO
Date: Oct.17, 2018


'''

from __future__ import absolute_import

import sys, csv, os
sys.path.append('..')

import numpy as np
from sklearn import metrics

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import settings

def getMetricDetail(y_truth, y_predict, is_hour_metric = True, is_display_detail = False):
    sc_all_r2= []
    sc_all_mse= []
    sc_all_ase= []
    sc_all_smape = []
    if is_hour_metric:
        print('Get hourly-performance score across all instances')
        for tm in xrange(settings.SEQ2INST_PARAM['predict_slot']):
            sc = metrics.r2_score(y_truth[:,tm], y_predict[:,tm])
            sc_all_r2.append(sc)

            sc2 = metrics.mean_squared_error(y_truth[:,tm], y_predict[:,tm])
            sc2 = np.sqrt(sc2)
            sc_all_mse.append(sc2)
            
            sc3 = metrics.mean_absolute_error(y_truth[:,tm], y_predict[:,tm])
            sc_all_ase.append(sc3)
            
            sc4 = smape(y_truth[:,tm], y_predict[:,tm])
            sc_all_smape.append(sc4)
            if is_display_detail:
                print('Hour, R2, RMSE, ABS, SMAPE: %d, %f, %f, %f, %f' % (tm+1, sc, sc2, sc3, sc4))
            
        print('Mean of hours, R2, RMSE, ABE, SMAPE: %f, %f, %f, %f' % (np.mean(sc_all_r2), np.mean(sc_all_mse), np.mean(sc_all_ase), np.mean(sc_all_smape)))
    else:
        print('Get instance-performance score across hours')
        for tm in xrange(y_truth.shape[0]):
            sc = metrics.r2_score(y_truth[tm, :], y_predict[tm, :])
            sc_all_r2.append(sc) #add negative so that all metric is lower is better

            sc2 = metrics.mean_squared_error(y_truth[tm, :], y_predict[tm, :])
            sc2 = np.sqrt(sc2)
            sc_all_mse.append(sc2)
            
            sc3 = metrics.mean_absolute_error(y_truth[tm, :], y_predict[tm, :])
            sc_all_ase.append(sc3)
            
            sc4 = smape(y_truth[tm, :], y_predict[tm, :])
            sc_all_smape.append(sc4)
            if is_display_detail:
                print('Single content, R2, RMSE, ABS, SMAPE: %d, %f, %f, %f, %f' % (tm+1, sc, sc2, sc3, sc4))
            
        print('Mean of all contents, R2, RMSE, ABE, SMAPE: %f, %f, %f, %f' % (np.mean(sc_all_r2), np.mean(sc_all_mse), np.mean(sc_all_ase), np.mean(sc_all_smape)))

    
    return {'r2':np.mean(sc_all_r2), 'rmse':np.mean(sc_all_mse), 'ase':np.mean(sc_all_ase), 'smape':np.mean(sc_all_smape), 'r2-72':sc_all_r2, 'smape-72':sc_all_smape, 'rmse-72': sc_all_mse, 'ase-72':sc_all_ase}

def getMetric(y_truth_i, y_predict_i, is_display_detail = False):
    tmpfile_tr = '/tmp/tempsxypairgggg_tr.test'
    tmpfile_pr = '/tmp/tempsxypairgggg_pr.test'
    tmpfile_diff = '/tmp/tempsxypairgggg_diff.test'
    y_truth = np.memmap(tmpfile_tr, dtype = 'float32', mode = 'w+', shape = (1,y_truth_i.shape[0] * y_truth_i.shape[1])) 
    y_predict = np.memmap(tmpfile_pr, dtype = 'float32', mode = 'w+', shape = (1,y_truth_i.shape[0] * y_truth_i.shape[1]))
    xy_diff = np.memmap(tmpfile_diff, dtype = 'float32', mode = 'w+', shape = (1,y_truth_i.shape[0] * y_truth_i.shape[1]))
    y_truth[:] = y_truth_i.flatten()[:]
    y_predict[:] = y_predict_i.flatten()[:]

    sc_all_r2,sc_all_mse,sc_all_ase,sc_all_smape  = r2_mse_ase_smape_score(y_truth, y_predict, xy_diff)

    os.remove(tmpfile_pr)
    os.remove(tmpfile_tr)
    os.remove(tmpfile_diff)

    score = {'r2': sc_all_r2,
            'rmse': sc_all_mse, 
            'ase': sc_all_ase, 
            'smape': sc_all_smape,
            'smape_inst': None, #sc_all_smape_inst.reshape(y_truth_i.shape),
            'ase_inst': None, #np.abs(y_truth - y_predict).reshape(y_truth_i.shape)
            }

    print('Mean of all contents, R2, RMSE, ABE, SMAPE: %f, %f, %f, %f' % (score['r2'], score['rmse'], score['ase'], score['smape']))
    
    return score

def r2_mse_ase_smape_score(ref, predict, xy_diff):
    tru_mean = np.mean(ref)
    xy_diff = ref - predict
    sm0 = np.sum(xy_diff*xy_diff)
    mse = np.sqrt(sm0 / xy_diff.size)
    ase = np.mean(np.abs(xy_diff))
    sc = 2.0 * np.mean(np.abs(xy_diff) / (ref + predict + 1.0e-20))
    
    xy_diff = ref - tru_mean
    sm1 = np.sum(xy_diff*xy_diff)
    r2 = 1. - sm0 / (sm1 + 1.0e-20)
    return r2, mse, ase, sc

def smape(ref, predict):
    if isinstance(ref, list):
        ref = np.array(ref)
    if isinstance(predict, list):
        predict = np.array(predict)
    ref = ref.flatten()
    predict = predict.flatten()
    
    #sc = np.zeros(shape=(ref.shape[0],1), dtype='float32')
    sc = 2.0 * np.abs(ref - predict) / (ref + predict + 1.0e-20)

    return np.mean(sc), None
