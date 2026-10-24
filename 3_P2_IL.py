#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 12:23:43 2018

@author: winstonlin
"""
import matplotlib
import matplotlib.pyplot as plt

import scipy.io
import numpy as np
import tensorflow as tf

import random
import pdb
import pywt
import time
import pickle
import csv

from sklearn import preprocessing
from scipy.stats import skew, kurtosis, zscore
from scipy.io import loadmat, savemat, wavfile
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from scipy.stats import spearmanr, pearsonr
from progressbar import *
from os.path import join
from os.path import sep as filesep

from keras import backend as K

from tensorflow.keras.layers import concatenate, Conv2D, Flatten, Conv1D
from tensorflow.keras.layers import Activation, Dense, Dropout, Reshape, LSTM
from tensorflow.keras.layers import ELU, LeakyReLU, ConvLSTM2D, Bidirectional
from tensorflow.keras.layers import BatchNormalization, TimeDistributed, GRU
from tensorflow.keras.activations import tanh
from tensorflow.keras.optimizers import SGD, Adam, Nadam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import model_from_json
from tensorflow.keras import Input, initializers, Model
from tensorflow.keras.callbacks import CSVLogger, LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.backend import expand_dims
from tensorflow.keras.models import load_model

random.seed(999)

##### #### #### #### #### #### #### 副程式 #### #### #### #### #### #### #### 
    
class CNNMDL:
    def __init__(self, MdName='test', Epoch=100, GpuSel=6, lra=0.001, LossFun='mean_squared_error', pre_trained_model_path = './model/pre_model/ssl_model.h5'):
    #def __init__(self, MdName = 'test', Epoch = 100, GpuSel = 6, lra = 0.001, LossFun = 'mean_squared_error'):
        self.MdNamePath=MdName
        self.Epoch=Epoch
        #self.GpuSel=str(GpuSel)
        self.GpuSel= str(6)#J gpu
        #self.LossFn='categorical_crossentropy'#J
        self.LossFn='binary_crossentropy'
        self.Init='TruncatedNormal' 
        self.pre_trained_model_path = pre_trained_model_path  # 新增读取预训练模型路径       
			
        self.learnset=Adam(lr=self._lr_schedule(0), beta_1=0.9, beta_2=0.999, epsilon = None, decay=0.0, amsgrad=False)
        #self.learnset=Adam(lr=self._lr_schedule(0), beta_1=0.9, beta_2=0.999, epsilon = 1e-8, decay=0.0, amsgrad=False)
        #self.learnset=SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        #SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.checkpointer = ModelCheckpoint(filepath=self.MdNamePath+".hdf5",
                                   monitor="val_loss",
                                   mode="min",
                                   verbose=1,
                                   save_best_only=True)        
    
        os.environ['CUDA_VISIBLE_DEVICES'] = self.GpuSel
        gpu_options=tf.compat.v1.GPUOptions(allow_growth=True,per_process_gpu_memory_fraction=0.9)
        sess=tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        tf.compat.v1.keras.backend.set_session(sess)

    def _lr_schedule(self, epoch):

        lr = 1e-4
        '''
        if epoch > 70: 
            lr *= 0.5e-3
        elif epoch > 50:
            lr *= 1e-3
        elif epoch > 30:
            lr *= 1e-2
        elif epoch > 10:
            lr *= 1e-1
        '''
        print('Learning rate: ', lr)

        return lr

    def build(self, InpTD, OutDim, LayerNodes):
        self.OutDim = OutDim  # 確保為 self.OutDim 賦值

        # 檢查是否有預訓練模型路徑
        if self.pre_trained_model_path:
            print(f"從 {self.pre_trained_model_path} 加載預訓練模型")
            #pdb.set_trace()
            self.CNNMDL = load_model(self.pre_trained_model_path, compile=False)
            print("成功加載預訓練模型。")

            # 只使用預訓練模型的前5層
            truncated_layers = [layer.output for layer in self.CNNMDL.layers[:5]]#self.CNNMDL.layers[:5]]
            truncated_model = Model(inputs=self.CNNMDL.input, outputs=truncated_layers[-1])
            
            # 添加新的輸出層用於分類任務
            # reshape_layer = Reshape((170,), name='reshape')(truncated_model.output)#1300ms
            reshape_layer = Reshape((380,), name='reshape')(truncated_model.output)#2691ms
            dense_1 = Dense(64, activation='tanh', kernel_initializer=self.Init, name='dense')(reshape_layer)
            dense_2 = Dense(32, activation='tanh', kernel_initializer=self.Init, name='dense_1')(dense_1)
            output_layer = Dense(2, activation='sigmoid', kernel_initializer=self.Init, name='dense_2')(dense_2)

            self.CNNMDL = Model(inputs=truncated_model.input, outputs=output_layer)
            
            self.CNNMDL.summary()
            #pdb.set_trace()
            with open(self.MdNamePath+".json", "w") as f:
                f.write(self.CNNMDL.to_json())

            
        
        else:
            # 如果沒有預訓練模型，則正常構建新模型
            wav_size = LayerNodes['time']
            ker_size = LayerNodes['kernel']
            fil_size = LayerNodes['filter']
            str_size = LayerNodes['strides']
            den_Nods = LayerNodes['dsel']

            self.MdlInp = Input(shape=(wav_size, InpTD, 1))

            for i in list(range(len(ker_size))):
                if i == 0:
                    self.HdMdl = Conv2D(ker_size[i], ker_size[i], strides=str_size[i], padding='same', data_format='channels_last', activation='tanh', kernel_initializer='TruncatedNormal')(self.MdlInp)
                else:
                    #HdMdl = Conv2D(ker_size[i], ker_size[i], strides=str_size[i], padding='same', data_format='channels_last', activation='tanh', kernel_initializer='TruncatedNormal')(HdMdl)
                    self.HdMdl = Conv2D(ker_size[i], ker_size[i], strides=str_size[i], padding='same', data_format='channels_last', activation='tanh', kernel_initializer='TruncatedNormal')(self.HdMdl)
            # 確保形狀在 Reshape 層之前是正確的
            print("HdMdl 輸出形狀:", self.HdMdl.shape)
            #pdb.set_trace()

            # 自動計算合適的形狀
            self.HdMdl = Reshape((-1,))(self.HdMdl)

            for i in list(range(len(den_Nods) - 1)):
                self.HdMdl = Dense(den_Nods[i], activation='tanh', kernel_initializer=self.Init)(self.HdMdl)

            self.HdMdl = Dense(den_Nods[len(den_Nods) - 1], activation='tanh', kernel_initializer=self.Init)(self.HdMdl)

            # 添加輸出層
            self.MdlOut = Dense(self.OutDim, activation='sigmoid', kernel_initializer=self.Init)(self.HdMdl)

            # 構建模型
            self.CNNMDL = Model(inputs=[self.MdlInp], outputs=[self.MdlOut])

        # 編譯模型
        self.CNNMDL.compile(loss='binary_crossentropy', optimizer=self.learnset)
        self.CNNMDL.summary()
        #pdb.set_trace()

    #def trainEEGCls(self,*Trh5_subs, *Vah5_subs, Cfg):
    #def trainEEGCls(self,TrEEGInfo_sub1, VaEEGInfo_sub1,TrEEGInfo_sub2,VaEEGInfo_sub2, TrEEGInfo_sub3, VaEEGInfo_sub3, Cfg):
    #def trainEEGCls(self,TrEEGInfo_sub1, VaEEGInfo_sub1,TrEEGInfo_sub2,VaEEGInfo_sub2, TrEEGInfo_sub3, VaEEGInfo_sub3,TrEEGInfo_sub4, VaEEGInfo_sub4,TrEEGInfo_sub5, VaEEGInfo_sub5,TrEEGInfo_sub6, VaEEGInfo_sub6,TrEEGInfo_sub7, VaEEGInfo_sub7, TrEEGInfo_sub8, VaEEGInfo_sub8,TrEEGInfo_sub9, VaEEGInfo_sub9, TrEEGInfo_sub10, VaEEGInfo_sub10, Cfg): 
    #def trainEEGCls(self,TrEEGInfo_sub1, VaEEGInfo_sub1,TrEEGInfo_sub2,VaEEGInfo_sub2, TrEEGInfo_sub3, VaEEGInfo_sub3,TrEEGInfo_sub4, VaEEGInfo_sub4,TrEEGInfo_sub5, VaEEGInfo_sub5,TrEEGInfo_sub6, VaEEGInfo_sub6, Cfg): 
    def trainEEGCls(self,
                TrEEGInfo_sub1, VaEEGInfo_sub1,
                TrEEGInfo_sub2, VaEEGInfo_sub2,
                TrEEGInfo_sub3, VaEEGInfo_sub3,
                TrEEGInfo_sub4, VaEEGInfo_sub4,
                TrEEGInfo_sub5, VaEEGInfo_sub5,
                TrEEGInfo_sub6, VaEEGInfo_sub6,
                TrEEGInfo_sub7, VaEEGInfo_sub7,
                TrEEGInfo_sub8, VaEEGInfo_sub8,
                TrEEGInfo_sub9, VaEEGInfo_sub9,
                TrEEGInfo_sub10, VaEEGInfo_sub10,
                TrEEGInfo_sub11, VaEEGInfo_sub11,
                TrEEGInfo_sub12, VaEEGInfo_sub12,
                TrEEGInfo_sub13, VaEEGInfo_sub13,
                TrEEGInfo_sub14, VaEEGInfo_sub14,
                TrEEGInfo_sub15, VaEEGInfo_sub15,
                TrEEGInfo_sub16, VaEEGInfo_sub16,
                TrEEGInfo_sub17, VaEEGInfo_sub17,
                TrEEGInfo_sub18, VaEEGInfo_sub18,
                TrEEGInfo_sub19, VaEEGInfo_sub19,
                TrEEGInfo_sub20, VaEEGInfo_sub20,
                TrEEGInfo_sub21, VaEEGInfo_sub21,
                TrEEGInfo_sub22, VaEEGInfo_sub22,
                TrEEGInfo_sub23, VaEEGInfo_sub23,
                TrEEGInfo_sub24, VaEEGInfo_sub24,
                TrEEGInfo_sub25, VaEEGInfo_sub25,
                TrEEGInfo_sub26, VaEEGInfo_sub26,
                TrEEGInfo_sub27, VaEEGInfo_sub27,
                TrEEGInfo_sub28, VaEEGInfo_sub28,
                TrEEGInfo_sub29, VaEEGInfo_sub29,
                TrEEGInfo_sub30, VaEEGInfo_sub30,
                TrEEGInfo_sub31, VaEEGInfo_sub31,
                TrEEGInfo_sub32, VaEEGInfo_sub32,
                TrEEGInfo_sub33, VaEEGInfo_sub33,
                TrEEGInfo_sub34, VaEEGInfo_sub34,
                TrEEGInfo_sub35, VaEEGInfo_sub35,
                TrEEGInfo_sub36, VaEEGInfo_sub36,
                TrEEGInfo_sub37, VaEEGInfo_sub37,
                TrEEGInfo_sub38, VaEEGInfo_sub38,
                TrEEGInfo_sub39, VaEEGInfo_sub39,
                TrEEGInfo_sub40, VaEEGInfo_sub40,
                TrEEGInfo_sub41, VaEEGInfo_sub41,
                TrEEGInfo_sub42, VaEEGInfo_sub42,
                TrEEGInfo_sub43, VaEEGInfo_sub43,
                TrEEGInfo_sub44, VaEEGInfo_sub44,
                TrEEGInfo_sub45, VaEEGInfo_sub45,
                TrEEGInfo_sub46, VaEEGInfo_sub46,
                TrEEGInfo_sub47, VaEEGInfo_sub47,
                TrEEGInfo_sub48, VaEEGInfo_sub48,
                TrEEGInfo_sub49, VaEEGInfo_sub49,
                TrEEGInfo_sub50, VaEEGInfo_sub50,
               Cfg):
    # def trainEEGCls(self,
    #              TrEEGInfo_sub1, VaEEGInfo_sub1,
    #              Cfg):
    
    #def trainEEGCls(self,TrEEGInfo, VaEEGInfo, Cfg, base_model):#J finetune  *Trh5_subs, *Vah5_subs, Cfg

    



        for sub_id in range(1, 51):  # 從 1 到 51
        # 動態生成 TrEEGInfo_sub 和 VaEEGInfo_sub 的變數名，假設這些變數已定義
            TrEEGInfo = eval(f'TrEEGInfo_sub{sub_id}')
            VaEEGInfo = eval(f'VaEEGInfo_sub{sub_id}')

        # 動態生成 g1_sub 和 g2_sub
            g1_sub = self._tr_Generator(TrEEGInfo, Cfg, True)
            g2_sub = self._va_Generator(VaEEGInfo, Cfg, True)

    # 動態命名的變數 g1_sub1, g1_sub2, 等
            locals()[f'g1_sub{sub_id}'] = g1_sub
            locals()[f'g2_sub{sub_id}'] = g2_sub            
        # g1_sub1 = self._tr_Generator(TrEEGInfo_sub1, Cfg, True)
        # g2_sub1 = self._va_Generator(VaEEGInfo_sub1, Cfg, True)
        


        #self.MdlOut = Dense(self.OutDim, activation = 'softmax', kernel_initializer = self.Init)(self.HdMdl)#J
        # self.MdlOut = Dense(self.OutDim, activation = 'sigmoid', kernel_initializer = self.Init)(self.HdMdl)

        # self.CNNMDL = Model(inputs=[self.MdlInp], outputs=[self.MdlOut]) 

        # 編譯模型 - 這裡確保已經存在 CNN 模型
        #J self.CNNMDL.compile(loss='categorical_crossentropy',optimizer=self.learnset)
        self.CNNMDL.compile(loss='binary_crossentropy',optimizer=self.learnset)
        #self.LossFn='binary_crossentropy'
        self.CNNMDL.summary()
        
        with open(self.MdNamePath+".json", "w") as f:
            f.write(self.CNNMDL.to_json())
#步骤4：for loop 训练模型

        for sub_id in range(1,51):  # 遍歷 sub_id 從 1 到 50
            # 假設 g1_sub 和 g2_sub 的變數已經存在，並按照 g1_sub1, g2_sub1 格式命名
            g1_sub = eval(f'g1_sub{sub_id}')
            g2_sub = eval(f'g2_sub{sub_id}')
            epochs = sub_id * 100#2,10 (50%),100(78%),700(?%)
    # 訓練模型，使用動態的 sub_id 作為數據
            self.Hist = self.CNNMDL.fit(
                x=g1_sub,
                validation_data=g2_sub,
                # epochs=self.Epoch,
                epochs=epochs,
                steps_per_epoch=Cfg['trbthperepoch'],
                validation_steps=Cfg['vabthperepoch'],
                callbacks=[self.checkpointer],
            )

    # 取得 loss 和 val_loss
        lossy = self.Hist.history['loss']  # 訓練損失
        val_lossy = self.Hist.history['val_loss']  # 驗證損失
        np_lossy = np.array(lossy).reshape((1, len(lossy)))
        np_val_lossy = np.array(val_lossy).reshape((1, len(val_lossy)))

    # 動態保存 loss 和 val_loss 文件，根據 sub_id 命名文件
        np.savetxt(f'loss_sub{sub_id}.txt', np_lossy)
        np.savetxt(f'val_loss_sub{sub_id}.txt', np_val_lossy)

        print(f"save loss for sub{sub_id}")
        print(f"save val_loss for sub{sub_id}")




################################################################################################
#步骤4：首次训练模型
#         self.Hist = self.CNNMDL.fit(x = g1_sub1,
#                                          validation_data = g2_sub1,
#                                          #epochs = self.Epoch,
#                                          epochs = 10,
#                                          steps_per_epoch=Cfg['trbthperepoch'],
#                                          validation_steps=Cfg['vabthperepoch'],
#                                          callbacks = [self.checkpointer],
#                                          )
#         lossy = self.Hist.history['loss']#J
#         val_lossy = self.Hist.history['val_loss']#J
#         np_lossy = np.array(lossy).reshape((1,len(lossy)))
#         np_val_lossy = np.array(val_lossy).reshape((1,len(val_lossy)))
#         np.savetxt('loss.txt',np_lossy)
#         np.savetxt('val_.txt',np_val_lossy)
#         print("save loss")
#         print("save val_loss")
   
        
    # 模型訓練
    def _tr_Generator(self, EEGInfoList, Cfg, Bool = False):

        UttLength =  Cfg['uttlength']# unit: second
        Fs = Cfg['fs'] # sample rate (pts/sec)
        NumRange = 2 # 1-10, 11-20
        NOISE_LEVEL = 0.01
   
        while True:
            SelUtt = np.random.randint(len(EEGInfoList), size = Cfg['trbthsize'])
    
            BthInpEEG = []
            for btInd in range(len(SelUtt)):
                EEGData, EEGAge, AgeLabel = self.EEGExtractor(EEGInfoList[SelUtt[btInd]], Cfg)

                if btInd == 0:
                    BthRefEEG = np.zeros((len(SelUtt), NumRange))
                BthRefEEG[btInd, AgeLabel] = 1
                
                while EEGData.shape[1] <= Fs * UttLength:
                    EEGData = np.concatenate((EEGData, EEGData), axis = 1)
                   
                fin_point = EEGData.shape[1] - Fs * UttLength + 1                   
                
                strPoints = np.random.randint(fin_point)
                fin_point = int(strPoints + Fs * UttLength)
    
                TpIpEEG = EEGData[:,strPoints:fin_point,...]
                if Cfg['norm']:
                    #pdb.set_trace()
                    #TpIpEEG = (TpIpEEG-np.min(TpIpEEG,axis=1))/(np.max(TpIpEEG,axis=1)-np.min(TpIpEEG,axis=1))
                    #J add noise here TpIpEEG.shape size = (1, 1000, 8)
                    add_noise = NOISE_LEVEL * np.random.randn(1, 1000, 8) #J 1000=> fs=200 5000=> fs=1000, 8->channel
                    TpIpEEG = (TpIpEEG - np.mean(TpIpEEG, axis = 1))/np.std(TpIpEEG, axis = 1)#J normalize 
                    #TpIpEEG = (TpIpEEG - np.mean(TpIpEEG, axis = 1))/np.std(TpIpEEG, axis = 1) + add_noise#J
                    
    
                if btInd == 0: 
                    BthInpEEG = TpIpEEG
                else:
                    BthInpEEG = np.concatenate((BthInpEEG, TpIpEEG), axis = 0)
                    
            yield np.array(BthInpEEG), np.array(BthRefEEG)

    def _va_Generator(self, EEGInfoList, Cfg, Bool = False):

        UttLength =  Cfg['uttlength']# unit: second
        Fs = Cfg['fs'] # sample rate (pts/sec)
        NumRange = 2 # 1-10, 11-20
   
        while True:
            SelUtt = np.random.randint(len(EEGInfoList), size = Cfg['trbthsize'])
    
            BthInpEEG = []
            for btInd in range(len(SelUtt)):
                EEGData, EEGAge, AgeLabel = self.EEGExtractor(EEGInfoList[SelUtt[btInd]], Cfg)

                if btInd == 0:
                    BthRefEEG = np.zeros((len(SelUtt), NumRange))
                BthRefEEG[btInd, AgeLabel] = 1
    
                while EEGData.shape[1] <= Fs * UttLength:
                    EEGData = np.concatenate((EEGData, EEGData), axis = 1)
                   
                fin_point = EEGData.shape[1] - Fs * UttLength + 1                   
                
                strPoints = np.random.randint(fin_point)
                fin_point = int(strPoints + Fs * UttLength)
    
                TpIpEEG = EEGData[:,strPoints:fin_point,...]
                if Cfg['norm']:
                    #TpIpEEG = (TpIpEEG-np.min(TpIpEEG,axis=1))/(np.max(TpIpEEG,axis=1)-np.min(TpIpEEG,axis=1))
                    TpIpEEG = (TpIpEEG - np.mean(TpIpEEG, axis = 1))/np.std(TpIpEEG, axis = 1)
    
                if btInd == 0: 
                    BthInpEEG = TpIpEEG
                else:
                    BthInpEEG = np.concatenate((BthInpEEG, TpIpEEG), axis = 0)
    
            yield np.array(BthInpEEG), np.array(BthRefEEG)
    

    # 讀模型存檔
    def LoadMdl(self, MdName = 'test'):

        with open(MdName+'.json', "r") as f:
            self.CNNMDL=model_from_json(f.read())

            self._LoadLSTMWeight(MdName)

    def _LoadLSTMWeight(self, MdName = 'test'):
        self.CNNMDL.load_weights(MdName+'.hdf5')

    # 資料預測
    def preditEEG(self,EEGInfoList, Cfg):
        BSize=10000
        #BSize=20
        
        UttLength =  Cfg['uttlength']# unit: second
        Fs = Cfg['fs'] # sample rate (pts/sec)
   
        EEGData, EEGRefr, AgeLabel = self.EEGExtractor(EEGInfoList, Cfg)
    
        while EEGData.shape[1] <= Fs * UttLength:
            EEGData = np.concatenate((EEGData, EEGData), axis = 1)
                   
        strPoints = 0
        #fin_point = EEGData.shape[1] - Fs * UttLength + 1
        fin_point = int(strPoints + Fs * UttLength)
    
        TpIpEEG = EEGData[:,strPoints:fin_point,...]
        
        if Cfg['norm']:
            #TpIpEEG = (TpIpEEG-np.min(TpIpEEG,axis=1))/(np.max(TpIpEEG,axis=1)-np.min(TpIpEEG,axis=1))
            TpIpEEG = (TpIpEEG - np.mean(TpIpEEG, axis = 1))/np.std(TpIpEEG, axis = 1)
                    
        PreRes = self.CNNMDL.predict(TpIpEEG, verbose=1, batch_size=BSize)
        PreRes = int(np.argmax(PreRes))
        
        return PreRes, AgeLabel    

    # 訊號處理
    def EEGExtractor(self, EEGInfo, Cfg):
    
        SelChanl = Cfg['eegchannel']

        TEEGTpDa = np.array(EEGInfo['data'][:, list(SelChanl)].astype('float32'))
        if Cfg['lowPass']:
           #pdb.set_trace()
           EEGTpDa = self.LFCExtract(TEEGTpDa,Cfg['dwtFun'],Cfg['dwtLevel'],Cfg['dBKpList'],Cfg['specbdonly'])
           # 有跑Cfg['lowPass']:
        else:
           #pdb.set_trace()#J check tim_points 687 
           EEGTpDa = TEEGTpDa
        tim_points, chanls = EEGTpDa.shape[0], len(Cfg['eegchannel'])#J
        

        return EEGTpDa.reshape(1, tim_points, chanls), EEGInfo['FilNme'].split(' ')[2].replace('Y','.').replace('y','.').replace('m.m00','').replace('m',''), int(EEGInfo['label'])
    
    def LFCExtract(self,InpData, wtFun, BdLevel = 2, BdList = [1,0,0], Return_spec_band_only = False):
        n_samples, channels = InpData.shape
        if not Return_spec_band_only:
            OutData = np.zeros((n_samples,channels))
        else:
            OutData = []
        for chInd in range(channels):
            band = pywt.wavedec(InpData[:,chInd].squeeze(), wtFun, level = BdLevel, mode = 'periodic')
            if not Return_spec_band_only:
                for bdInd in range(BdLevel+1):
                    if not BdList[bdInd]:
                        band[bdInd] = np.zeros(band[bdInd].shape)
                ncA = pywt.waverec(band, wtFun, mode = 'periodic')
                OutData[:,chInd] = ncA[:n_samples]
            else:
                for bdInd in range(BdLevel+1):
                    if BdList[bdInd]:
                        OutData.append(band[bdInd])
                                     
        if Return_spec_band_only:
            OutData = np.array(OutData).T
        return OutData
        

##### #### #### #### #### #### #### 主程式 #### #### #### #### #### #### #### 
#### 參數設定
lra=1e-4#1e-4
Epoch=200#J 1000

#Front:[AF3,AF4,F1,Fz,F2,Fp1,Fpz,Fp2],
#EEGChannel = [4,5,9,10,11,1,2,3]
#EEGChannel = [4,5,9,10,11,1,2,3,16,17,25,26,34,35,8,44] #front + left
#EEGChannel = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 24, 25]  # front + left (scalp map)#
#Front:[AF3,AF4,F1,Fz,F2,Fp1,Fpz,Fp2],
#EEGChannel = [4,5,9,10,11,1,2,3]
#辛博提出的alpha band 16 channels
EEGChannel = [9,17,18,19,26,27,28,35,36,37,44,45,46,52,53,54]#

InpTD = len(EEGChannel)


#model name
MdNme='EEG_CNN_inc_'
MdNme += '50subs'

SavMdlPth = join('model', MdNme)
SavCsvPth = join('csv', MdNme + '.csv')

Trh5_subs = []
Vah5_subs = []

# 使用 for 循環來遍歷 sub_id 從 1 到 6
for sub_id in range(1, 51):#51
    # 加載對應的 Training_sub 文件
    with open(f'Training_sub{sub_id:02d}.p', 'rb') as f:
        Trh5_subs.append(pickle.load(f))  # 將加載的數據存入 Trh5_subs 列表

    # 加載對應的 Validation_sub 文件
    with open(f'Validation_sub{sub_id:02d}.p', 'rb') as f:
        Vah5_subs.append(pickle.load(f))  # 將加載的數據存入 Vah5_subs 列表


# with open('Training_sub1.p', 'rb') as f:
#     Trh5_sub_test = pickle.load(f)
# with open('Validation_sub1.p', 'rb') as f:
#     Vah5_sub1 = pickle.load(f)
    

    
Cfg = {'vabthsize': 1,#1
        'trbthsize': 1,#1
        'eegchannel': EEGChannel,
        'uttlength': 3,     # second 3
        'fs': 200,#J
        'trbthperepoch': 10,#100
        'vabthperepoch': len(Trh5_subs),
        'lowPass': True,
        'dwtFun': 'db10',
        'dwtLevel': 2,#2       # 跟著 dBKpList這個設定，如dBKpList有N個值，則dwtLevel為N-1
        #'dBKpList': [1,0,0,0],
        'dBKpList': [1,0,0], # 可設定保留的頻帶(1︰留；0︰不留)︰ex. [1,0,0]︰留 0~25 Hz。[1,1,0]︰留 0~50 Hz。[1,0,0,0]︰留 0~12.5 Hz。[0,1]︰留 50~100 Hz。
        'specbdonly':True,
        'norm':False#J default False
        }


#J for 16 channel
LayerNodes = {'time': int(Cfg['uttlength'] * Cfg['fs']),
              'kernel': [10,10,10,10],
              'filter': [(200,4), (100,4), (50,4)],
              'strides': [(2,2), (2,2), (2,2),(2,2)],
              'dsel': [64, 32]
            }




#### 模型訓練
cnneeg=CNNMDL(SavMdlPth, Epoch, 0)  # LSTM model
cnneeg.build(InpTD, 2, LayerNodes)
# 如已有模型，可註解下面這一行 用 # 這個符號
cnneeg.trainEEGCls(*Trh5_subs, *Vah5_subs, Cfg)


# 讀取模型存檔 
cnneeg.LoadMdl(SavMdlPth)

#模型測試#################################################################################################################################
##for all

for sub_id in range(1, 51):  # 從 1 到 50
        # 動態生成文件名
    file_name = f'Testing_sub{sub_id:02d}.p'

    # 加載對應的pickle文件
    with open(file_name, 'rb') as f:
        Tsh5_sub = pickle.load(f)  # 加載pickle數據到 Tsh5_sub

    Pred_EEG_Age = []
    GdTu_EEG_Age = []
    Num = []  # number
    Nme = []
    Age = []
    Dis = []

    for eegInd in Tsh5_sub:
        PredAge, EEGRefr = cnneeg.preditEEG(eegInd, Cfg)
        Num.append(eegInd['FilNme'].split(' ')[0])  # 001
        Nme.append(eegInd['FilNme'].split(' ')[1])  # 女聲
        Age.append(eegInd['FilNme'].split(' ')[2])  # 6y11m
        Pred_EEG_Age.append(PredAge)
        GdTu_EEG_Age.append(EEGRefr)

    print(f"Sub ID {sub_id}:")
    print("Pred_EEG_Age:", Pred_EEG_Age)
    print("GdTu_EEG_Age:", GdTu_EEG_Age)

    # 測試結果寫入 csv檔 (excel) : SavCsvPth
    Pred_EEG_Age.insert(0, 'Pred_EEG_Age')
    GdTu_EEG_Age.insert(0, 'GdTu_EEG_Age')
    Num.insert(0, ' ')
    Nme.insert(0, ' ')
    Age.insert(0, ' ')
    Dis.insert(0, ' ')

    # 指定每個 sub_id 的保存路徑
    SavCsvPth = f"results_sub_{sub_id}.csv"

    with open(SavCsvPth, 'w', encoding='utf-8-sig', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(Num)  # 001
        writer.writerow(Nme)  # 女聲
        writer.writerow(Age)  # 6y11m
        writer.writerow(Dis)  # epilepsy
        writer.writerow(Pred_EEG_Age)
        writer.writerow(GdTu_EEG_Age)




