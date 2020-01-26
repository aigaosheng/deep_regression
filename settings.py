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


#--------------------------------
# Setting for feature & model training
#

#for filtering noisy data
MIN_OBSERVED_PV_SLOT = 1 #if observed PV slot is smaller than it, discard the record. In Mediacorp's case, PV record is 1 hour interval. So slot is hour
PV_PREDICTION_SLOT = 2#72 #maximal future predicted ages

#define noisy record type and the record number found in dataset
PVRECORDTYPE = {'clean': 0, #which is used for pv prediction
                'missPubtime': 0, #record miss publication time 
                'oldContent': 0,  #too long time when first PV is recorded since publication time
                'fewPvSlot': 0, #too few pv slots in record < min_observed_pv_slot, in the pv_train_record_window
                'longInactive': 0 #long time inactive between the sequential time interval, in the pv_train_record_window
}

#set category of PV
RV_CATEGORY = {'RV': 'vv'}
TAG_SCHEMA = { #key: output field-name, value: current input table field name
}

TAG_SCHEMA_REV = {}
for k, v in TAG_SCHEMA.iteritems():
    TAG_SCHEMA_REV[v] = k 

#set control parameters for extract training instances from radio view (RV) sequence
lookAheadHour_rv = 23

'''
RV model
'''
SEQ2INST_PARAM_RV = {
    'lookback_slot': lookAheadHour_rv, #3, #5,, #how many previous PV slots are used for prediction. Actual input PV is lookback+ reference slot, i.e. pv vector is lookback_slot+1 dimension
    'predict_slot': 72, #how many future PV slots to be predicted

    #set constant variable for preprocessing feature
    'dayweek': 7, #day of week 
    'hourday': 24, #hour of day
    'value_avoid_overflow': 1., #1.0, #add the value to PV to avoid overflow in case of log(pv)
    'padding_value': 0., #pad the PV value before first PV record
    'is_adjust_pv1': False, #true if adjust first PV based on its time difference from publication time

    # --- Feature selection & normalization control --- #
    #for processing publication day/hour
    'is_pubtime_dayhour_feature': False, #use pubtime as a feature
    'is_log_dayhour': True, # True: log day and hour, False: norm to [0,1]

    #for processing PV day/hour/age
    'is_pv_age_as_feature': False, #True, #True: use age of latest PV in the feature as feature
    'is_pv_dayhour_as_feature': False, #True, #True: use PV record day/hour as feature

    'is_log_pv_x': True, # when True, use log-domain pv value for input. 
    'is_log_pv_y': True, # when True, use log-domain pv value for target. 
    'is_delta': False, #True is to use delta PV as feature

    'is_site_as_feature': False, #encoding site as a feature

    #set age to be used to choose best model in training stage
    'is_section_as_feature': False, #True, use the section as a feature

    #for video particularly
    'is_videoDuration_feature': False, #True, 
    # --- Feature selection & normalization END  --- #

    'is_republish_ignored': False, #True, #True, ignore republish article, ie contentID with pub-time more than 1-hour later than first PV time

    'cutoff-age': None, #if pv1 is more than 10 ages, as noisy
    'inst_age_threshold': None, #if None, use all. Otherwise, only use instance less than it

}


DEFAULT_REPUBLISH_AGE = 1

#encoding sites
SITE_NAMES_RV = {
    'a': 0,
    'b': 1,
}

SITE_NAMES_TO_MODEL_MAP_RV = { #define property name to model name map to support new property site can use old model

}


#set this for training case to choose specific setting. in online testing, pv/vv is different module, so import its own parameters
SITE_NAMES = SITE_NAMES_RV
lookAheadHour = lookAheadHour_rv
SEQ2INST_PARAM = SEQ2INST_PARAM_RV
SITE_NAMES_MODEL_MAP = SITE_NAMES_TO_MODEL_MAP_RV


CUT_OFF_AGE = {
    'train': None, #training, cutoff articles pv1 is more than 20-hours older
    'test': None, #test, all used
}

REMOVE_REPUBLISH = {
    'train': True,
    'test': True, #test, all used
}

#defin feature preprocessing
FEATURE_PREPROCESS_CONFIG = {
    'isGaussScale': False, #normalize vector to Gaussian(mean=0, std=1)
    'isGaussScaleX': False, #normalize vector to Gaussian(mean=0, std=1)
    'isGaussScaleY': False, #normalize vector to Gaussian(mean=0, std=1)

}

#define decision tree config
IS_CONFIG_TREE = True #true to use the following config for tree regression
TREE_CONFIG = {
    'split_ratio': 1.0 / 100.,
    'leaf_ratio': 1. / 200.
}
#note: if optimizing smape, test on elee, smap reduce 7%, but r2 reduce too much, from ~0.05-> -0.012
MODEL_SELECTION_METRIC = 'smape' #'r2' # ' #rmse, 

#setting field names of memmap meta-info
MEMMAP_IOFILE = 'iofile' #input file and output memmap file name pair
MEMMAP_X_DIM = 'x_dim'   #pv/vv feature dimension
MEMMAP_Y_DIM = 'y_dim'   #predicted pv/vv dimension 
MEMMAP_X_META_DIM = 'x_meta_dim'    #meta feature dimension,
MEMMAP_INSTS_NUM = 'inst_counter'   #total instances in memmap file
MEMMAP_DIM = 'dim' #sum of x_dim * number-of-source, y_dim and x_meta_dim
MEMMAP_SOURCE_N = 'n_source' #source of PV

#setting index-location for x_feat_meta
META_PUB_DAYOFWEEK = 0
META_PUB_HOUROFDAY = 1
META_PV_REF_AGE = 2
META_PV_REF_DAYOFWEEK = 3
META_PV_REF_HOUROFDAY = 4
META_ARTICLE_SITE = 5
META_USED_DIM = SEQ2INST_PARAM['is_pubtime_dayhour_feature'] * 2 + SEQ2INST_PARAM['is_pv_age_as_feature'] + SEQ2INST_PARAM['is_pv_dayhour_as_feature'] * 2 #2 #5 #3

print('****** Number of meta-feature used %d *****' % META_USED_DIM)

#set a threshold to prune training instances which sum of 72-hour target pv/vv smaller than it
MIN_PV_VV_Y = 0
MIN_PV_VV_X = 1 #max(int(float(lookAheadHour) * 0.1), 1)#3
IS_REMOVE_LOW_PV_VV = True #False #for training
IS_REMOVE_LOW_PV_VV_TEST = True #False #False  #for testing
latest_hour_vv_filter = True

#DNN
REDUCE_SAMPLE_SIZE_RATIO = None #0.5 #None, #0.5 #for toggle, too big sample subset. For cna, no needed
IS_SAMPLE_WEIGHT_USED = False

DNN_CONFIG_RV = {
        'model_type': 'dnn', #'dnn', #'embed'
        'meta_layer_dim': 64, #[64]*2+[32], 256, #512, #set layer size for meta feature
        'layer_stack': [512, 512, 256, 128, 64], #[64] * 4,
        'is_batch_norm': True, #if True, use batch_normalize
        'dropout_rate': 0.1, #0.1, #if None, not use dropout. other must be value in 0.0-1.0
        'i_dim': 0,
        'o_dim': SEQ2INST_PARAM['predict_slot'],
        'learning_rate': 0.005 * 4,
        'n_epoch': 30,
        'n_batch_size': 32,
        'activation_i': 'relu', #tanh', #'relu', #internal layer
        'activation_o': 'relu', #'relu', #internal layer
        'n_internal_layer': 4, #4, #number of dense internal layers
        'is_continue_train': False, #True: if a model already exists in the model-path fold, start train from the model as intialization. False: train from random intialization 
        'validation_split': 0.1,
        'loss': 'smapeLoss', #mse', #smapeLoss', #'mean_squared_logarithmic_error', #'mse', #mean_squared_error', #mean_squared_logarithmic_error', #mse',
        'val_metric': 'smapeMetric',
        #for meta embedding
        'seq_length': 3, #1, only support 3 or 1
        'embed_dim': 64, #output dimension of embedded dim
        'n_lstm_dim': 64, #cell size of LSTM
    }

DNN_CONFIG = DNN_CONFIG_RV
IS_DEV_SET_PROVIDED = False #True #if have seperate DEV set, set it True. Otherwise, use a portion of train as DEV in DNN training

#set for checkpoint, if there is best model in current model path, will continue training from this model. empty folder if training from intialization
#CHECKPOINT_MODEL_NAME = '-{epoch:02d}-{val_loss:.4f}.hdf5' #set checkpoint model extened name
CHECKPOINT_MODEL_NAME = '-{epoch:02d}-{val_%s:.4f}.hdf5' % DNN_CONFIG['val_metric'] #set checkpoint model extened name
CHECKPOINT_MODEL_MONITOR = 'val_%s' % DNN_CONFIG['val_metric'] #'val_loss'
CHECKPOINT_MODEL_VERBOSE = 1
CHECKPOINT_MODEL_SAVE_BEST_ONLY = True
CHECKPOINT_MODEL_MODE = 'min'
CHECKPOINT_MODEL_PERIOD = 1
REMOVE_CHECKPOINT_MODEL = True #if true, remove checkpoint models after best model is chosen and saved

#set for checkpoint, if there is best model in current model path, will continue training from this model. empty folder if training from intialization
#CHECKPOINT_MODEL_NAME = '-{epoch:02d}-{val_loss:.4f}.hdf5' #set checkpoint model extened name
CHECKPOINT_MODEL_NAME = '-{epoch:02d}-{val_%s:.4f}.hdf5' % DNN_CONFIG['val_metric'] #set checkpoint model extened name
CHECKPOINT_MODEL_MONITOR = 'val_%s' % DNN_CONFIG['val_metric'] #'val_loss'
CHECKPOINT_MODEL_VERBOSE = 1
CHECKPOINT_MODEL_SAVE_BEST_ONLY = True
CHECKPOINT_MODEL_MODE = 'min'
CHECKPOINT_MODEL_PERIOD = 1
REMOVE_CHECKPOINT_MODEL = True #if true, remove checkpoint models after best model is chosen and saved
#
EMBED_AGE_MAX = 24* 10
if DNN_CONFIG['model_type'] == 'embed':
    if DNN_CONFIG['seq_length'] == 1:
        #combine age, pub-day-hour(168), reference-day-hour(168) to single code
        #EMBED_DIM_MAX = EMBED_AGE_MAX * SEQ2INST_PARAM['dayweek'] * SEQ2INST_PARAM['hourday'] * SEQ2INST_PARAM['dayweek'] * SEQ2INST_PARAM['hourday']
        EMBED_DIM_MAX = 1 * SEQ2INST_PARAM['dayweek'] * SEQ2INST_PARAM['hourday'] * SEQ2INST_PARAM['dayweek'] * SEQ2INST_PARAM['hourday']
        print('INFO: **** Embed age/pub-day-hour/reference-day-hour into 1 code. Currently ignore age because of model memory ****')
    elif DNN_CONFIG['seq_length'] == 3:
        EMBED_DIM_MAX = EMBED_AGE_MAX + SEQ2INST_PARAM['dayweek'] * SEQ2INST_PARAM['hourday'] + SEQ2INST_PARAM['dayweek'] * SEQ2INST_PARAM['hourday']
        print('INFO: **** Invidually embed age/pub-day-hour/reference-day-hour *** ')
    else:
        raise ValueError('DNN_CONFIG["seq_length"] must be 1 or 3')
else:
    EMBED_DIM_MAX = 1




