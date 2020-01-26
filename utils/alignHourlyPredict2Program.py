#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The module is to align 72-hour prediction of MeRadio to program.
The module is to re-aggregate hourly data into program based on program schedule

Author: SHENG GAO
Date: Aug.15, 2018
"""
from __future__ import absolute_import

import os, sys
import datetime
import csv, re
import argparse

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#sys.path.append('/home/gao/Work/view_forcast')


def loadRadioProgram(radio_property_file = 'radio.properties'):
    '''
    radio_property_file: radio program schedule (text file), e.g. each line such as
            95-0-0-4=The Best Mix of Music
            95-0-5-9=Muttons In The Morning
            ....
    
    return: dict of 24hr radio program schedule, index by: radio_sattion: {weekday/weekendday-[0-23]: program_name/id}, e.g.
            {'95':{'0-3':program_code_internal}}
            program_to_internal_code: {'95_The Best Mix of Music': program_code_internal}
      
    '''
    program_internal_code2name = {} #key: program name, value: program_id
    radio_station_program_schedule = {} #key: radio station ID, value: dict(0-23, prgram_id)
    with open(radio_property_file) as i_fpro:
        program_code_start = 0
        for rln in i_fpro:
            if rln.startswith('#'):
                continue
            try:
                rt, pname = rln.split('=')
            except:
                continue
            radioname, dayid, tm1, tm2 = rt.split('-')
            pname = pname.lower().strip()
            #get program code
            pname_idx = radioname + '_' + pname
            try:
                internal_code = program_internal_code2name[pname_idx]
            except:
                program_code_start += 1
                internal_code = program_code_start
                program_internal_code2name[pname_idx] = internal_code
            #
            if radioname not in radio_station_program_schedule:
                radio_station_program_schedule[radioname] = {}
            #refer to program ID: radio_station_id / hour-key (weekday: 0-program_id, weekend: 5/6-program_id)
            for k in xrange(int(tm1), int(tm2)+1):
                #index by weekday/weekend day + '-' + hour
                if dayid == '0':
                    #week day Mon-Fri
                    for wk in xrange(6):
                        prom_key = getProgramKey(wk, k) #'d' + str(wk) + 'h' + str(k)
                        radio_station_program_schedule[radioname][prom_key] = internal_code
                else:
                    #weekend day: 5 (sat), 6 (sunday)
                    prom_key = getProgramKey(int(dayid)-1, k) #'d' + str(int(dayid)-1) + 'h' + str(k)
                    radio_station_program_schedule[radioname][prom_key] = internal_code

    return radio_station_program_schedule, program_internal_code2name

def getProgramKey(day_id, hr_id):
    return 'd' + str(day_id) + 'h' + str(hr_id)

def align72hrPredictToProgram(radio_station, batch_time, pred, radio_station_program_schedule):
    '''
    radio_station: MeRadio station name, e.g. '95', '933', etc
    batch_time: the latest timestamp of input views, i.e. predict 72-hour hereafter
    pred: 72-hour prediction integer value
    radio_station_program_schedule: program schedule by day-hour for each station

    return: filter 72-hour pred to get corresponding 72-hour predict value for specific program of radio_station at batch_time

    '''
    if not isinstance(batch_time, datetime.datetime):
        raise Exception('batch_time must be instance of datatime ')
    prom_key = getProgramKey(batch_time.weekday(), batch_time.hour)
    prom_code = radio_station_program_schedule[radio_station][prom_key]
    pred_prom = [0] * len(pred)
    for h, v in enumerate(pred):
        cur_tm = batch_time + datetime.timedelta(hours = h + 1)
        cur_key = getProgramKey(cur_tm.weekday(), cur_tm.hour)
        if radio_station_program_schedule[radio_station][cur_key] == prom_code:
            #pred_prom[h] = (cur_tm,v)
            pred_prom[h] = v

    return pred_prom

#test
#a, b= loadRadioProgram()
