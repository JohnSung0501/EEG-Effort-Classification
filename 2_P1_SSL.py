#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Reshape, Input, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pickle
import pdb


EEGChannel = [9,17,18,19,26,27,28,35,36,37,44,45,46,52,53,54]#


Cfg = {
    'eegchannel': EEGChannel,
    'input_shape': (600, len(EEGChannel), 1),  
    'batch_size': 1,  # 32
    'epochs': 3000,
    'mask_ratio': 0.3
}

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def mask_input(data, mask_ratio=0.3):
    mask = np.random.rand(*data.shape) < mask_ratio
    masked_data = data.copy()
    masked_data[mask] = 0
    return masked_data

def pad_data(data, target_length):
    """
    对数据进行零填充，如果长度不足 target_length，则在时间维度上用零填充。
    """
    current_length = data.shape[1]
    if current_length >= target_length:
        return data[:, :target_length, :]  
    else:

        padding_length = target_length - current_length

        zero_padding = np.zeros((data.shape[0], padding_length, data.shape[2]))

        padded_data = np.concatenate((data, zero_padding), axis=1)
        return padded_data

def create_ssl_model(input_shape):

    input_layer = Input(shape=input_shape)

    x = Conv2D(10, (10, 10), strides=(2, 2), activation='tanh', padding='same', kernel_initializer='TruncatedNormal', bias_initializer='zeros')(input_layer)
    x = Conv2D(10, (10, 10), strides=(2, 2), activation='tanh', padding='same', kernel_initializer='TruncatedNormal', bias_initializer='zeros')(x)
    x = Conv2D(10, (10, 10), strides=(2, 2), activation='tanh', padding='same', kernel_initializer='TruncatedNormal', bias_initializer='zeros')(x)
    x = Conv2D(10, (10, 10), strides=(2, 2), activation='tanh', padding='same', kernel_initializer='TruncatedNormal', bias_initializer='zeros')(x)



    #x = Reshape((170,))(x)#1.3s
    x = Reshape((380,))(x)#2.69s


    flatten = Flatten()(x)
    encoded_representation = Dense(64, activation='tanh', kernel_initializer='TruncatedNormal')(flatten)
    encoded_representation = Dense(32, activation='tanh', kernel_initializer='TruncatedNormal')(encoded_representation)

    reconstruction = Dense(np.prod(input_shape), activation='tanh')(encoded_representation)
    reconstruction_output = Reshape(input_shape, name='reconstruction')(reconstruction)

    ssl_model = Model(inputs=input_layer, outputs=reconstruction_output)

    ssl_model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='mean_squared_error',  # 
        metrics=['mse']
    )
    
    return ssl_model

def load_data(train_file, valid_file, eeg_channels):
    with open(train_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(valid_file, 'rb') as f:
        valid_data = pickle.load(f)


    train_x = np.array([sample['data'][:, eeg_channels] for sample in train_data])  
    valid_x = np.array([sample['data'][:, eeg_channels] for sample in valid_data])  

    train_x = train_x[:, ::5, :]
    valid_x = valid_x[:, ::5, :]

    print(f"Train data shape after downsampling: {train_x.shape}")
    print(f"Valid data shape after downsampling: {valid_x.shape}")
    
     
    train_x = pad_data(train_x, target_length=Cfg['input_shape'][0])
    valid_x = pad_data(valid_x, target_length=Cfg['input_shape'][0])

    print(f"Train data shape after padding: {train_x.shape}")
    print(f"Valid data shape after padding: {valid_x.shape}")
    

    train_x = (train_x - np.mean(train_x, axis=(1, 2), keepdims=True)) / np.std(train_x, axis=(1, 2), keepdims=True)
    valid_x = (valid_x - np.mean(valid_x, axis=(1, 2), keepdims=True)) / np.std(valid_x, axis=(1, 2), keepdims=True)


    train_x = train_x.reshape((-1, train_x.shape[1], train_x.shape[2], 1))
    valid_x = valid_x.reshape((-1, valid_x.shape[1], valid_x.shape[2], 1))


    masked_train_x = mask_input(train_x, mask_ratio=Cfg['mask_ratio'])
    masked_valid_x = mask_input(valid_x, mask_ratio=Cfg['mask_ratio'])

    return masked_train_x, train_x, masked_valid_x, valid_x


def main():

    # load data
    Trh5_subs = []
    Vah5_subs = []
    for sub_id in range(1, 101):#1~100
        train_file = f'Training_sub{sub_id:02d}.p'
        valid_file = f'Validation_sub{sub_id:02d}.p'

    with open(train_file, 'rb') as f:
        Trh5_subs.append(pickle.load(f))  # 

    # 加載對應的 Validation_sub 文件
    with open(valid_file, 'rb') as f:
        Vah5_subs.append(pickle.load(f))  # 

    # train_file = 'Training_sub1.p'
    # valid_file = 'Validation_sub1.p'
    masked_train_x, train_x, masked_valid_x, valid_x = load_data(train_file, valid_file, Cfg['eegchannel'])




    model = create_ssl_model(input_shape=Cfg['input_shape'])
    model.summary()
    pdb.set_trace()


    model.fit(
        x=masked_train_x,
        y=train_x,  # 
        epochs=Cfg['epochs'],
        batch_size=Cfg['batch_size'],
        validation_data=(masked_valid_x, valid_x)
    )


    loss, mse = model.evaluate(
        x=masked_valid_x,
        y=valid_x
    )
    print(f"Test Loss: {loss}, Mean Squared Error: {mse}")


    model.save('ssl_model.h5')
    print("Model saved as ssl_model.h5")


if __name__ == "__main__":
    main()
