#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
The module is for live page/video view prediction. Due to the reason of update history of system, here some variables are still use prefix vv because first version of DNN is built for video view forecasting, and then merge previos PV system (scikit tree) into it. The same reason for  the model name prefix is still pv_model.pkl because PV prediction system is first developed.

Author: Sheng Gao
Date: Oct. 17, 2018
'''
from __future__ import absolute_import

import sys
import os
import csv
import cPickle as pickle
import numpy as np
import random
import datetime
import json
import argparse
import glob
import logging
from logging.config import fileConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from keras.models import load_model

from meradio.featureExtractor import contentMetaFeature, onlineFeatureExtractor
from model.dnnModel import predictReport as predictReportDnn
from model.dnnModel import addUserDefinedMetric
from model.model import predictReport as predictReportTree
from utils.clusterInst import clusterInstances, getInstTopicKey
import settings
import configEnv

from utils.alignHourlyPredict2Program import loadRadioProgram, align72hrPredictToProgram


class predictService(object):
    '''
    The class is to wrap loading models & prediction given segment
    model resources directory structure:
        1. --root:
        2.    --Site_name
        3.        --model_type
        4.            --model_file
        model_file_name is same for all Site & model_type
    '''
    def __init__(self, model_path, model_type = 'retree-m_7-24-inst', topic_method = '7-24-inst', model_file_name = 'pv_model.pkl'):
        self.__model_path = model_path
        self.__vv_model_set = {} #save all models here. indexed by site name
        self.__vvSnap = {} #temparary vv snap
        self.__model_type = model_type
        self.__model_name = model_file_name
        self.__topic_model_type = topic_method
        self.__featureModelSet = None
        #load models
        try:
            self.__loadVvModel()
        except:
            raise ValueError('Warning: error when loading VV models from %s' % self.__model_path)

        #load radio property/program schedule
        self.radio_schedule, self.radio_internal_program_code = loadRadioProgram(configEnv.radio_poperty_file)
        #define log file
        logging.basicConfig(filename=configEnv.logging_file,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%Y %H:%M:%S',
                            level=logging.WARNING)        
        self.mev_logger = logging.getLogger()


    def __loadVvModel(self):
        '''
        Load model
        '''

        print('Start to loading prediction models')
        count_down = len(settings.SITE_NAMES.keys())
        #for site_name in ['meradio']: #settings.SITE_NAMES.keys():
        for site_name, _ in settings.SITE_NAMES.iteritems():
            model_file = os.path.join(self.__model_path , site_name, configEnv.RESULT_FOLD, self.__model_type, self.__model_name)
            print('INFO: Loading %s model from %s' % (site_name, model_file))
            if self.__model_type == 'dnn':
                model_files = glob.glob(model_file + '.*')
                addUserDefinedMetric()
                meModel = {}
                for fl in model_files:
                    gkey = fl.split('.')[-1]
                    meModel[gkey] = load_model(fl)
                self.__vv_model_set[site_name] = meModel
                

            else:
                self.mev_logger.warning('Warning: only support DNN')
                raise Exception('Warning: only support DNN')
                '''with open(model_file, 'rb') as i_fmodel:
                    print('%d Loading %s from %s' % (count_down, site_name, model_file))
                    self.__vv_model_set[site_name] = pickle.load(i_fmodel)
                    count_down -= 1
                    try:
                        self.__featureModelSet = pickle.load(i_fmodel)
                    except:
                        self.__featureModelSet = None'''


    def predict(self, vv_snapshot):
        '''
        Predict 72-hour vv based on vv_snapshot.
        'site':, 'contentID', 'pubtime', 'vvtimeFirst', 'vvtimeLatest', 'vv'
        '''
        try:    
            site_name = settings.SITE_NAMES_MODEL_MAP[vv_snapshot['site']]                
        except:
            self.mev_logger.warning('Warning: site field %s' % vv_snapshot['site'])
            raise ValueError('Warning: missing site name in input. field name: site')

        #print('********** HHHHH ')
        try:    
            contentid = vv_snapshot['contentID']
        except: 
            self.mev_logger.warning('Warning: contentID field %s' % vv_snapshot['contentID'])
            raise ValueError('Warning: missing contentID in input. field name: contentID')        
        try:
            lastestVvTime_str = vv_snapshot['mevtimeLatest'].strip()
            lastestVvTime = datetime.datetime.strptime(lastestVvTime_str, '%Y-%m-%d %H:%M:%S')  
        except:
            self.mev_logger.warning('Warning: mevtimeLatest %s (e.g. 2019-02-11 22:10:30)' % vv_snapshot['mevtimeLatest'])
            raise ValueError('Warning: missing latest RV timestamp (to predict hereafter 72-hour from this timestamp) in input or wrong format. field name: vvtimeLatest, format: Y-M-D H:M:S')
        try:
            vv_str = vv_snapshot['mev'].strip()
            #print(vv_str)
            vv_seq = np.array([int(v) for v in vv_str.split(',')])
            history_vv = [vv_seq]
        except:
            self.mev_logger.warning('Warning: mev filed value %s' % vv_snapshot['mev'])
            raise ValueError('Warning: missing PV sequence, a string with seperator <,> between values, e.g., PV string: "10, 20, 30", field name: rv')
        '''
        try:
            vvs_str = vv_snapshot['smev'].strip()
            vvs_seq = np.array([int(v) for v in vvs_str.split(',')])
            history_vv.append(vvs_seq)
        except:
            pass
            #raise ValueError('Warning: missing Social PV sequence, a string with seperator <,> between values, e.g., PV string: "10, 20, 30", field name: vv')
        '''
        content_meta = []
        pubtm = None
        for k, vvsrc in enumerate(history_vv):
            meta_info = contentMetaFeature(contentid, site_name, pubtm, lastestVvTime, vvsrc, is_republish_ignored = False)
            content_meta += [meta_info]
            history_vv[k] = meta_info['adjust_pvsnap']

        test_insts_x_meta_embed, test_insts_x,  test_insts_x_meta= onlineFeatureExtractor(contentid, site_name, pubtm, lastestVvTime, history_vv, content_meta)

        #test_inst_index = clusterInstances(test_insts_x_meta, test_insts_x)
        test_inst_index = getInstTopicKey(test_insts_x_meta, method = self.__topic_model_type)
        #print(history_vv)
        #print(test_insts_x)
        #print(test_inst_index)

        if self.__model_type == 'dnn':
            y_Predict = predictReportDnn(test_insts_x, test_inst_index, self.__vv_model_set[site_name], self.__model_type, self.__topic_model_type, self.__featureModelSet, test_insts_x_meta_embed)
        else:
            y_Predict = predictReportTree(test_insts_x, test_inst_index, self.__vv_model_set[site_name], self.__model_type, self.__topic_model_type, self.__featureModelSet)

        ss = ''
        latest_72hr_predict = {'contentID': contentid, 'mevtimeLatest': lastestVvTime_str, 'mev72h': {}}

        if vv_snapshot['site'] != 'podcast':
            try:
                #print(y_Predict[0])
                self.mev_logger.warning('%s '% str(y_Predict[0]))
                pred_program = align72hrPredictToProgram(vv_snapshot['site'], lastestVvTime, y_Predict[0], self.radio_schedule)
                #print(pred_program)
            except:
                self.mev_logger.warning('Warning: align 72hr prediction to program error %s' % str(vv_snapshot))
                pred_program = [0] * len(y_Predict[0])
        else:
            pred_program = y_Predict[0]

        for k, v in enumerate(pred_program):
            datetime_ref = lastestVvTime + datetime.timedelta(hours = k + 1)
            ss += str(datetime_ref.hour) + ':' + str(int(v)) + ' '
            latest_72hr_predict['mev72h'][k] = int(v)
        #print(ss)
        return latest_72hr_predict
