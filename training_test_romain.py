#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 18:03:35 2019

@author: dimitri.hamzaoui
"""

from generators import Quadriview_DataGenerator
from models import BN_quadriview_model

import tensorflow as tf
import keras.backend as K
from model_utils import test_noise_characs, test_slice, model_display, ROC_curve, summary_model_training
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, History, EarlyStopping, LearningRateScheduler
import os


    

generator_train = Quadriview_DataGenerator(csv_file = "dataset_train.csv", batch_size = 64, normalize="brain", transfo=False,
                                               replace_option=True, prefix="/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/", mask_files = "/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/masks/")
generator_train.metadata;
    
generator_val = Quadriview_DataGenerator(csv_file = "dataset_val.csv", batch_size = 64, normalize="brain", transfo=False,
                                                   replace_option=True, prefix="/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/", mask_files = "/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/masks/")
generator_val.metadata;
    
#config = tf.ConfigProto(device_count = {'GPU': 1 , 'CPU': 12})
config = tf.ConfigProto(log_device_placement=True)
#config.gpu_options.per_process_gpu_memory_fraction = 0.25
sess = tf.Session(config=config) 
K.set_session(sess)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

print('toto')
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
print('tzata')


model = BN_quadriview_model(name = "BN_on_brain")
optimizer = Adam(lr = 5e-5, beta_1=0.9, beta_2=0.999)
early_stop = EarlyStopping(monitor='val_loss', patience=8)

hist = History()
if not os.path.isdir("/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/NN_saved/Q_BN_brain"):
        os.mkdir("/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/NN_saved/Q_BN_brain")
saveoint = ModelCheckpoint(filepath="/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/NN_saved/Q_BN_brain/Q_weights_{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1, save_best_only=False)
model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics=['accuracy'])
model.fit_generator(generator_train, epochs = 100, validation_data=generator_val, verbose=1,
                    use_multiprocessing = True, workers = 12, callbacks = [hist, early_stop, saveoint])


summary_model_training(model, prefix="/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/")
#model_eval(model, "dataset_test.csv", prefix="/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/")
#model_display(model, csv_file = "dataset_test.csv", prefix="/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/", batch_size = 10)
#test_noise_characs(model, "dataset_test.csv", prefix="/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/")
#test_slice(model, "dataset_test.csv", prefix="/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/")
#ROC_curve(model, prefix="/network/lustre/iss01/cenir/analyse/irm/users/dimitri.hamzaoui/", batch_size = 512, csv_file = "dataset_test.csv")    
sess.close()
del sess
