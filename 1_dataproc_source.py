#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 12:23:43 2018

@author: winstonlin
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


import scipy.io
import numpy as np

import random
import pdb
import re
import pickle
import os

from os.path import join
from os.path import sep as filesep
from scipy.signal import butter, lfilter

random.seed(999)

##### #### #### #### #### #### #### function #### #### #### #### #### #### #### 

def readEEGFolder(InpPath, label=0, Tot_Class=3, NmeMode=False):

    if os.path.isdir(InpPath):
        FilList=os.listdir(InpPath)
        EEGData=[]
        for filename in sorted(FilList):
            with open(join(InpPath,filename)) as f:
                EEGData.append(EEGParRead(f.readline().strip()))
                EEGData[-1].update(dict([('ChannelName',f.readline().strip())]))
                if NmeMode:
                    EEGData[-1].update(EEGFileName(re.sub('\s+',' ',filename)))
                else:
                    EEGData[-1].update(dict([('FilNme',re.sub('\s+',' ',filename))]))

                Data=np.zeros((int(EEGData[-1]['TimePoints']),int(EEGData[-1]['Channels'])))
                lineInd=0
                for lineData in f.readlines():
                    Data[lineInd,:]=np.array([float(i) for i in re.sub('\s+',' ',lineData.strip()).split(' ')])
                    lineInd+=1
                #pdb.set_trace()
                fs = 1000  # 
                lowcut =  7 # 
                highcut = 13.0  # 
                num_channels = 60
                num_samples = 2691
                eeg_data = Data
                filtered_data = np.zeros_like(eeg_data)
                for ch in range(num_channels):
                    filtered_data[:, ch] = bandpass_filter(eeg_data[:, ch], lowcut, highcut, fs, order=5)
                #pdb.set_trace()
                #EEGData[-1].update(dict([('data',Data)]))#raw data
                EEGData[-1].update(dict([('data',filtered_data)]))#add BPF
                EEGData[-1].update(dict([('label',label)]))
                #pdb.set_trace()

    return EEGData

def butter_bandpass(lowcut, highcut, fs, order=5):#order = 5
    nyq = 0.5 * fs  # 
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
def EEGFileName(InpFileName):
    Header={}
    StrList=InpFileName.split(' ')
    Header.update(dict([('Name',StrList[0])]))
    Header.update(dict([('Age',float(StrList[1][0:-1]))]))
    for par in list(range(2,len(StrList))):
        if StrList[par].split('.')[0].upper() == 'awake'.upper():
            Header.update(dict([('Status',StrList[par]+'_'+StrList[par+1])]))
        elif StrList[par].split('.')[0].upper() == 'sleep'.upper():
            Header.update(dict([('Status',StrList[par])]))

    return Header

def EEGParRead(InpString):
    EEGHeader={}
    StrList=InpString.split(' ')
    for par in StrList:
        EEGHeader.update(dict([(par.split('=')[0],par.split('=')[1])]))

    return EEGHeader

def get_filepaths(directory):

    OutList = []
    for root, directories, files in os.walk(directory):
        filepath = os.path.join(root, filename)
        OutList.append(filepath)

    return OutList

##### #### #### #### #### #### #### main #### #### #### #### #### #### #### 



# Loop through sub_id from 01 to 50
for sub_id in range(1, 101):#J 1~100
    # Format sub_id to be two digits (e.g., '01', '02', ..., '50')
    sub_id_str = f'{sub_id:02}'
    #sub_id_str = str(sub_id)

    # Paths using the variable sub_id_str
    TrainPath1_sub = f'/data2/user_johnsung/EEG_data/100P_file/Linux_data_T100/train/EEG_data1_1p/T{sub_id_str}'
    TrData_sub_1 = readEEGFolder(TrainPath1_sub, label=0)
    TrainPath2_sub = f'/data2/user_johnsung/EEG_data/100P_file/Linux_data_T100/train/EEG_data2_1p/T{sub_id_str}'
    TrData_sub_2 = readEEGFolder(TrainPath2_sub, label=1)

    # Combine the training data into a list
    Train_list_sub = []
    for inData_sub in [TrData_sub_1, TrData_sub_2]:
        for DataList_sub in inData_sub:
            Train_list_sub.append(DataList_sub)
    #pdb.set_trace()
    # Testing paths
    TestPath1_sub = f'/data2/user_johnsung/EEG_data/100P_file/Linux_data_T100/test/EEG_data1_1p/T{sub_id_str}'
    TsData_1_sub = readEEGFolder(TestPath1_sub, label=0)
    TestPath2_sub = f'/data2/user_johnsung/EEG_data/100P_file/Linux_data_T100/test/EEG_data2_1p/T{sub_id_str}'
    TsData_2_sub = readEEGFolder(TestPath2_sub, label=1)

    # Combine the test data into a list
    Test_list_sub = []
    for inData_sub in [TsData_1_sub, TsData_2_sub]:
        for DataList_sub in inData_sub:
            Test_list_sub.append(DataList_sub)

    # Validation data is the same as test data
    Valid_list_sub = Test_list_sub

    # Save the data to pickle files for each sub_id
    with open(f'Training_sub{sub_id_str}.p', 'wb') as f:
        pickle.dump(Train_list_sub, f)
    with open(f'Validation_sub{sub_id_str}.p', 'wb') as f:
        pickle.dump(Valid_list_sub, f)
    with open(f'Testing_sub{sub_id_str}.p', 'wb') as f:
        pickle.dump(Test_list_sub, f)



