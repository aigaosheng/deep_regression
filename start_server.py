#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This is server call
'''
from __future__ import absolute_import

import sys, os
#sys.path.append('..')

import json
import time, datetime
from flask import Flask,  jsonify, request
from flask_restful import Api, Resource
import numpy as np

import settings
import configEnv
from server.predictor import predictService


#vvseq = ','.join(['1'] * (settings.lookAheadHour + 1))
ssss='2.8903718 2.9957323 1.9459101 2.9957323 3.295837 3.5553482 3.0445225 1.7917595 2.4849067 2.7725887 2.0794415 2.1972246 3.5263605 2.7725887 2.1972246 2.9957323 1.609438 0.0 1.9459101 0.0 1.7917595 1.609438 1.9459101 2.0794415'
vvseq2 = map(lambda x:int(np.exp(float(x)) + 0.5)-1, ssss.split())
vvseq = ','.join(map(lambda x:str(x), vvseq2))
args={"contentID": 122, "mev": vvseq, "mevtimeLatest": "2019-03-11 0:00:00", "site": "95"}

my_predictor = predictService(configEnv.ROOT_PATH, model_type = configEnv.MODEL_TYPE, topic_method = configEnv.TOPIC_METHOD, model_file_name = configEnv.MODEL_NAME)

#the following must added for keras.for each site model, must call once. Don't know why. Maybe keras bug. If not add, report tensor shape do not match   
if configEnv.MODEL_TYPE == 'dnn':
    for site_name in settings.SITE_NAMES.keys(): 
        args_site = {}
        args_site.update(args)
        args_site['site'] = site_name
        try:
            results = my_predictor.predict(args_site)
        except:
            pass
else:
    results = my_predictor.predict(args)
#
forcastServiceApp = Flask(__name__)
forcastServiceApi = Api(forcastServiceApp)


# shows a list of all todos, and lets you POST to add new tasks
class predictApi(Resource):
    def get(self):
        return 'NO data' #pvset

    def post(self):
        i_vv_history = request.get_json(force=True)
        if not isinstance(i_vv_history, list):
            i_vv_history = [i_vv_history]
        #print(i_vv_history)
        pred_results = []
        #start = time.time()

        #aggregate past 24-hour radio station to view sequence from program level data. Only for MeRadio
        #because Meradio model is trained based on radio hourly streaming views rather than program. But UI need program level
        if i_vv_history[0]['site'] != 'podcast':
            i_vv_history, i_vv_delay = buildMeradioInputFromProgram(i_vv_history)
            for ivv_feat, delay in zip(i_vv_history, i_vv_delay):
                ivv_result = my_predictor.predict(ivv_feat)
                shift72 = {}

                for kk in xrange(len(ivv_result['mev72h']) - delay):
                    shift72[kk] = ivv_result['mev72h'][kk + delay]
                for kk in xrange(len(ivv_result['mev72h']) - delay, len(ivv_result['mev72h'])):
                    shift72[kk] = 0                
                ivv_result['mev72h'] = shift72
                pred_results.append(ivv_result)
        else:
            for ivv_feat in i_vv_history:
                ivv_result = my_predictor.predict(ivv_feat)
                pred_results.append(ivv_result)
        pred_response = forcastServiceApp.response_class(
            response=json.dumps(pred_results),
            status=200,
            mimetype='application/json'
        )
        '''
        end = time.time()
        print('*** Running time*****')
        print(end - start)
        print('**** END ****')
        '''

        return pred_response #json.dumps(pred_results), 201

def buildMeradioInputFromProgramOrdered(i_radio_program_history):
    #assume the list of each call is ordered by time from old to latest
    i_radio_view = [i_radio_program_history[0]]
    sum_vv_seq = [int(v) for v in i_radio_view[0]['mev'].strip().split(',')]
    for k in xrange(1, len(i_radio_program_history)):
        i_prog = i_radio_program_history[k]
        prog_seq = [int(v) for v in i_prog['mev'].strip().split(',')]
        sum_vv_seq = map(lambda x,y:x+y, sum_vv_seq, prog_seq)
        i_prog['mev'] = ','.join([str(v) for v in sum_vv_seq])
        i_radio_view.append(i_prog)
    return i_radio_view

def buildMeradioInputFromProgram(i_radio_program_history):
    #assume the list of each call is any order, so need to sort from old to latest
    latest_hour_pos = []
    for param in i_radio_program_history:
        vv = param['mev'].strip().split(',')
        for k in xrange(len(vv)-1, -1, -1):
            if vv[k] != '0':
                latest_hour_pos.append(k)
                break
    sorted_call_id = [vv[0] for vv in sorted(enumerate(latest_hour_pos), key = lambda x:x[1])] 
    # 
    i_radio_view = [] #[i_radio_program_history[sorted_call_id[0]]]
    i_radio_view_delay = []
    #sum_vv_seq = [int(v) for v in i_radio_view[0]['mev'].strip().split(',')]
    #
    sum_vv_seq = None
    for k in xrange(0, len(i_radio_program_history)):
        i_prog = i_radio_program_history[sorted_call_id[k]]
        prog_seq = [int(v) for v in i_prog['mev'].strip().split(',')]
        if k > 0:
            sum_vv_seq = map(lambda x,y:x+y, sum_vv_seq, prog_seq)
        else:
            sum_vv_seq = prog_seq
        #remove latest zeros
        sum_vv_seq2 = [0] * (len(sum_vv_seq) - latest_hour_pos[sorted_call_id[k]]-1) + sum_vv_seq[:latest_hour_pos[sorted_call_id[k]]+1]
        i_prog['mev'] = ','.join([str(v) for v in sum_vv_seq2])
        #update timestamp to its real
        lastestVvTime_str = i_prog['mevtimeLatest'].strip()
        lastestVvTime = datetime.datetime.strptime(lastestVvTime_str, '%Y-%m-%d %H:%M:%S')  

        actual_time = lastestVvTime - datetime.timedelta(hours = len(sum_vv_seq) - latest_hour_pos[sorted_call_id[k]]-1)
        i_prog['mevtimeLatest'] = str(actual_time)
        i_radio_view.append(i_prog)
        i_radio_view_delay.append(len(sum_vv_seq) - latest_hour_pos[sorted_call_id[k]]-1)
    #restore input order
    i_radio_view2 = [None] * len(i_radio_view)
    i_radio_view_delay2 = [None] * len(i_radio_view)
    for k0, k in enumerate(sorted_call_id):
        i_radio_view2[k0] = i_radio_view[k]
        i_radio_view_delay2[k0] = i_radio_view_delay[k]
    return i_radio_view2, i_radio_view_delay2

#buildMeradioInputFromProgram([{"site":"95", "contentID":"122", "pubtime":"2019-3-21 10:00:00", "mevtimeLatest":"2019-3-21 12:00:00", "mev":"0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2651,2347,18050"},{"site":"95", "contentID":"122", "pubtime":"2019-3-21 10:00:00", "mevtimeLatest":"2019-3-21 12:00:00", "mev":"0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3076,1983,5673,6394,8508,0,0,0"},{"site":"95", "contentID":"122", "pubtime":"2019-3-21 10:00:00", "mevtimeLatest":"2019-3-21 12:00:00", "mev":"0,0,0,0,0,0,0,0,0,0,0,621,309,429,402,234,0,0,0,0,0,0,0,0"}])

# Setup the PV-predictor Api resource routing
forcastServiceApi.add_resource(predictApi, '/predictmev')


if __name__ == '__main__':
    '''
    curl http://localhost:5000/predict -d '{"site":"elle", "contentID":'122', "pubtime":"2016-07-24 18:00:00","pvtimeLatest":"2018-04-02 13:00:00", "pv":"0,1,2,3"}' -X POST --header 'Content-Type: application/json'  --header 'Accept: application/json'    

    curl http://localhost:5000/predict -d '{"site":"cna", "contentID":'122', "pubtime":"2016-07-24 18:00:00","vvtimeLatest":"2018-04-02 13:00:00", "vv":"0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3"}' -X POST --header 'Content-Type: application/json'  --header 'Accept: application/json'    

    
    curl http://localhost:5000/predict -d '[{"site":"cna", "contentID":'122', "pubtime":"2016-07-24 18:00:00","pvtimeLatest":"2018-04-02 15:00:00", "pv":"0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4"}, {"site":"toggle", "contentID":"122", "pubtime":"2016-07-24 18:00:00", "pvtimeLatest":"2018-04-02 13:00:00", "pv":"0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4"}]' -X POST --header 'Content-Type: application/json'  --header 'Accept: application/json'

    curl http://localhost:5000/predictmev -d '[{"site":"meradio", "contentID":'122', "pubtime":"2016-07-24 18:00:00","mevtimeLatest":"2018-04-02 15:00:00", "mev":"0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3"}, {"site":"podcast", "contentID":"122", "pubtime":"2016-07-24 18:00:00", "mevtimeLatest":"2018-04-02 13:00:00", "mev":"0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,7"}]' -X POST --header 'Content-Type: application/json'  --header 'Accept: application/json'

    curl http://localhost:5000/predictmev -d '{"site":"podcast", "contentID":"122", "pubtime":"2016-07-24 18:00:00", "mevtimeLatest":"2018-04-02 13:00:00", "mev":"0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,7"}' -X POST --header 'Content-Type: application/json'  --header 'Accept: application/json'

    '''
    #global my_pv_predictor
    #my_pv_predictor = vvPredictor.vvPredictor(ROOT_PATH, model_type = MODEL_TYPE, topic_method = TOPIC_METHOD, model_file_name = MODEL_NAME)
    
    #results = my_pv_predictor.predict(args_cna)
    #results = my_pv_predictor.predict(args_toggle)
    #print(results)

    forcastServiceApp.run(debug=False, threaded=True, host='0.0.0.0', port=5200)
