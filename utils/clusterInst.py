#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This is to get grid/cluster index for each instance

Author: SHENG GAO
Date: Oct.1, 2018


'''

from __future__ import absolute_import

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import settings

def clusterInstances(inst_meta, X, Y = None, method = None):
    '''
    clustering instances into topic/cluster based meta-dat or data-driven method
    To avoid the data sparity issue, e.g. some topics has no instances if based on meta-data clustering. Always train a generic model based on all instances

    Input:
        inst_meta: instance meta feature, each row is a instance, (0,1):pub-day/hour, (3,4): reference day-hour
        X: feature, each row vector is a instance
        Y: target, each row vector is a instance's target
        method: the method to group instances into topic

    return:
        a dictionary to store instance index for each topic
    '''
    if method == '7-24':
        print('INFO: cluster instances based on publication day of week and hour of day')
        inst_index = dict()
        #cluster instance based publication date/time
        for n_day in xrange(settings.SEQ2INST_PARAM['dayweek']):
            for n_hour in xrange(settings.SEQ2INST_PARAM['hourday']):
                inst_index[keyOfDayHour(n_day, n_hour)] = []
        for ik in xrange(inst_meta.shape[0]):
            n_day, n_hour = inst_meta[ik,[0,1]]
            try:
                inst_index[keyOfDayHour(n_day, n_hour)].append(ik)
            except:
                inst_index[keyOfDayHour(n_day, n_hour)] = [ik]
        inst_index['generic'] = range(inst_meta.shape[0])
    elif method == '7-24-inst':
        print('INFO: cluster instances based on day of week and hour of day of each instance')
        inst_index = dict()
        #cluster instance based publication date/time
        for n_day in xrange(settings.SEQ2INST_PARAM['dayweek']):
            for n_hour in xrange(settings.SEQ2INST_PARAM['hourday']):
                inst_index[keyOfDayHour(n_day, n_hour)] = []
        for ik in xrange(inst_meta.shape[0]):
            n_day, n_hour = inst_meta[ik, [3,4]]
            try:
                inst_index[keyOfDayHour(n_day, n_hour)].append(ik)
            except:
                inst_index[keyOfDayHour(n_day, n_hour)] = [ik]
        inst_index['generic'] = range(inst_meta.shape[0])
    elif method == '24-inst':
        print('INFO: cluster instances based on day of week and hour of day of each instance')
        inst_index = dict()
        #cluster instance based publication date/time
        for n_hour in xrange(settings.SEQ2INST_PARAM['hourday']):
            inst_index[keyOfDayHour(None, n_hour)] = []
        for ik in xrange(inst_meta.shape[0]):
            n_day, n_hour = inst_meta[ik, [3,4]]
            try:
                inst_index[keyOfDayHour(None, n_hour)].append(ik)
            except:
                inst_index[keyOfDayHour(None, n_hour)] = [ik]
        inst_index['generic'] = range(inst_meta.shape[0])
    elif method == '7-inst':
        print('INFO: cluster instances based on day of week and hour of day of each instance')
        inst_index = dict()
        #cluster instance based publication date/time
        for n_day in xrange(settings.SEQ2INST_PARAM['dayweek']):
            inst_index[keyOfDayHour(n_day, None)] = []
        for ik in xrange(inst_meta.shape[0]):
            n_day, n_hour = inst_meta[ik, [3,4]]
            try:
                inst_index[keyOfDayHour(n_day, None)].append(ik)
            except:
                inst_index[keyOfDayHour(n_day, None)] = [ik]
        inst_index['generic'] = range(inst_meta.shape[0])
    elif method == None:
        inst_index = dict()
        inst_index['generic'] = range(inst_meta.shape[0])
    else:
        raise Exception('topic method error')
    
    
    return inst_index

def getInstTopicKey(inst_meta, method = None):
    '''
    clustering instances into topic/cluster based meta-dat or data-driven method
    To avoid the data sparity issue, e.g. some topics has no instances if based on meta-data clustering. Always train a generic model based on all instances

    Input:
        inst_meta: instance meta feature, each row is a instance
        X: feature, each row vector is a instance
        Y: target, each row vector is a instance's target
        method: the method to group instances into topic

    return:
        a dictionary to store instance index for each topic
    '''
    ik = 0
    if method == '7-24':
        n_day, n_hour = inst_meta[ik,[0,1]]
        return keyOfDayHour(n_day, n_hour)
    elif method == '7-24-inst':
        n_day, n_hour = inst_meta[ik, [3,4]]
        return keyOfDayHour(n_day, n_hour)
    elif method == '24-inst':
        n_day, n_hour = inst_meta[ik, [3,4]]
        return keyOfDayHour(None, n_hour)
    elif method == '7-inst':
        n_day, n_hour = inst_meta[ik, [3,4]]
        return keyOfDayHour(n_day, None)
    elif method is None:
        return 'generic'
    else:
        raise Exception('topic method error')
    
def getInstTopicKey22(inst_meta, method = None):
    '''
    clustering instances into topic/cluster based meta-dat or data-driven method
    To avoid the data sparity issue, e.g. some topics has no instances if based on meta-data clustering. Always train a generic model based on all instances

    Input:
        inst_meta: instance meta feature, each row is a instance
        X: feature, each row vector is a instance
        Y: target, each row vector is a instance's target
        method: the method to group instances into topic

    return:
        a dictionary to store instance index for each topic
    '''
    keylist = []
    if method == '7-24':
        for ik in range(inst_meta.shape[0]):
            n_day, n_hour = inst_meta[ik,[0,1]]
            keylist.append(keyOfDayHour(n_day, n_hour))
    elif method == '7-24-inst':
        for ik in range(inst_meta.shape[0]):
            n_day, n_hour = inst_meta[ik, [3,4]]
            keylist.append(keyOfDayHour(n_day, n_hour))
    elif method == '24-inst':
        for ik in range(inst_meta.shape[0]):
            n_day, n_hour = inst_meta[ik, [3,4]]
            keylist.append(keyOfDayHour(None, n_hour))
    elif method == '7-inst':
        for ik in range(inst_meta.shape[0]):
            n_day, n_hour = inst_meta[ik, [3,4]]
            keylist.append(keyOfDayHour(n_day, None))
    elif method is None:
        return ['generic'] * inst_meta.shape[0]
    else:
        raise Exception('topic method error')

def keyOfDayHour(n_day = None, n_hour = None):
    '''
    The function is to generate mapping key for input
    '''
    if n_day != None:
        dstr = str(int(n_day))
    else:
        dstr = ''
    if n_hour != None:
        hstr = str(int(n_hour))
    else:
        hstr = ''
    return 'd'+dstr+'h'+hstr
    
