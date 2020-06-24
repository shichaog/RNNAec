#!/usr/bin/env python

from __future__ import print_function

import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.layers import concatenate
from keras import losses
from keras import regularizers
from keras.constraints import min_max_norm
from keras.callbacks import LearningRateScheduler
import h5py

from keras.constraints import Constraint
from keras import backend as K
import numpy as np

import tensorflow as tf

import os
import sys


#with tf.Session(config=tf.ConfigProto(device_count = {'GPU':0, 'CPU':20}, inter_op_parallelism_threads=1, intra_op_parallelism_threads=20, log_device_placement=True)) as sess:
#    K.set_session(sess)
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.compat.v1.ConfigProto(device_count = {'GPU':0, 'CPU':20}, inter_op_parallelism_threads=1, intra_op_parallelism_threads=20, log_device_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.42
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

lr_base = 0.001
epochs = 430
lr_power = 0.9
def lr_scheduler(epoch):
    '''if lr_dict.has_key(epoch):
        lr = lr_dict[epoch]
        print 'lr: %f' % lr'''


    mode='power_decay'
    if mode is 'power_decay':
        # original lr scheduler
        lr = lr_base * ((1 - float(epoch) / epochs) ** lr_power)
    if mode is 'exp_decay':
        # exponential decay
        lr = (float(lr_base) ** float(lr_power)) ** float(epoch + 1)
    # adam default lr
    if mode is 'adam':
        lr = 0.001

    if mode is 'progressive_drops':
        # drops as progression proceeds, good for sgd
        if epoch > 0.9 * epochs:
            lr = 0.0001
        elif epoch > 0.75 * epochs:
            lr = 0.001
        elif epoch > 0.5 * epochs:
            lr = 0.01
        else:
            lr = 0.1

    if (isinstance(lr, complex)):
        lr = 0.000001
    if (lr <= 0.0000001):
        lr = 0.0000001
    print('lr: %f' % lr)
    
    return lr

scheduler = LearningRateScheduler(lr_scheduler)

def my_crossentropy(y_true, y_pred):
    return K.mean(2*K.abs(y_true-0.5) * K.binary_crossentropy(y_pred, y_true), axis=-1)

def mymask(y_true):
    return K.minimum(y_true+1., 1.)

def msse(y_true, y_pred):
    return K.mean(mymask(y_true) * K.square(K.sqrt(y_pred) - K.sqrt(y_true)), axis=-1)

def mycost(y_true, y_pred):
    return K.mean(mymask(y_true) * (10*K.square(K.square(K.sqrt(y_pred) - K.sqrt(y_true))) + K.square(K.sqrt(y_pred) - K.sqrt(y_true)) + 0.01*K.binary_crossentropy(y_pred, y_true)), axis=-1)

def my_accuracy(y_true, y_pred):
    return K.mean(2*K.abs(y_true-0.5) * K.equal(y_true, K.round(y_pred)), axis=-1)

class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
            'c': self.c}

reg = 0.000001
constraint = WeightClip(0.499)

print('Build model...')
main_input = Input(shape=(None, 112), name='main_input')
tmp = Dense(24, activation='tanh', name='input_dense', kernel_constraint=constraint, bias_constraint=constraint)(main_input)
vad_gru = GRU(32, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, name='vad_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(tmp)
vad_gru2 = GRU(24, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, name='vad_gru2', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(vad_gru)
delay_output = Dense(1, activation='sigmoid', name='delay_output', kernel_constraint=constraint, bias_constraint=constraint)(vad_gru2)
noise_input = keras.layers.concatenate([tmp, vad_gru, main_input])
noise_gru = GRU(48, activation='relu', recurrent_activation='sigmoid', return_sequences=True, name='noise_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(noise_input)
denoise_input = keras.layers.concatenate([vad_gru2, noise_gru, main_input])

aec_gru = GRU(96, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, name='aec_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(denoise_input)

aec_output = Dense(36, activation='sigmoid', name='aec_output', kernel_constraint=constraint, bias_constraint=constraint)(aec_gru)

model = Model(inputs=main_input, outputs=[aec_output, delay_output])

model.compile(loss=[mycost, my_crossentropy],
              metrics=[msse],
              optimizer='adam', loss_weights=[10, 0.1])


batch_size = 128  # 128+64 for gpu version, original version is 32 for cpu

print('Loading data...')
if len(sys.argv) == 1:
    print("Usage:{0} model_in.h5\n  Note that default output is mid_weights.hdf5 and final_weights.hdf5".format(sys.argv[0]))
    exit(1)
# with h5py.File('/share/tmp/training.h5', 'r') as hf:
# with h5py.File('training.h5', 'r') as hf:
with h5py.File(sys.argv[1], 'r') as hf:
    all_data = hf['data'][:]
print("load {} successful...".format(sys.argv[1]))

window_size = 2000

nb_sequences = len(all_data)//window_size
print(nb_sequences, ' sequences')
x_train = all_data[:nb_sequences*window_size, :112]
x_train = np.reshape(x_train, (nb_sequences, window_size, 112))

y_train = np.copy(all_data[:nb_sequences*window_size, 112:148])
y_train = np.reshape(y_train, (nb_sequences, window_size, 36))

noise_train = np.copy(all_data[:nb_sequences*window_size, 148:184])
noise_train = np.reshape(noise_train, (nb_sequences, window_size, 36))

vad_train = np.copy(all_data[:nb_sequences*window_size, 184:185])
vad_train = np.reshape(vad_train, (nb_sequences, window_size, 1))

delay_train = np.copy(all_data[:nb_sequences*window_size, 185:186])
delay_train = np.reshape(vad_train, (nb_sequences, window_size, 1))

all_data = 0
#x_train = x_train.astype('float32')
#y_train = y_train.astype('float32')

print(len(x_train), 'train sequences. x shape =', x_train.shape, 'y shape = ', y_train.shape)

print('Train...')

midWeightPath = './mid_weights.hdf5'
endEpochHookCB = keras.callbacks.ModelCheckpoint( midWeightPath,
                monitor = 'val_loss',
                save_weights_only = True,
                verbose = 1,
                save_best_only = False,
                period = 1 )

if os.path.exists(midWeightPath):
    model.load_weights( midWeightPath )
    print("checkpoint_loaded")

model.fit( x_train, [ y_train, delay_train ],
          batch_size = batch_size,
          epochs = 430,
          initial_epoch = 0,
          validation_split = 0.00,
          callbacks = [ endEpochHookCB ] )
          #callbacks = [ endEpochHookCB, scheduler ] )
model.save("./final_weights.hdf5")
