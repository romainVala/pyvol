"""
Created on Wed May 15 18:03:35 2019

@author: dimitri.hamzaoui
"""
import sys
sys.path.append('./Documents/Python_files')
sys.path.append('./tf_cnnvis')
from generators import Quadriview_DataGenerator
from models import BN_quadriview_model

import tensorflow as tf
import keras.backend as K
from model_utils import test_noise_characs, test_slice, model_display, ROC_curve
from keras.optimizers import Adam
import tf_cnnvis.tf_cnnvis





    
if __name__ == "__main__":
    generator = Quadriview_DataGenerator(csv_file = "dataset_test.csv", batch_size = 64, normalize="global", transfo=False,
                                               replace_option=True)
    generator.metadata;
    
    config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 24} ) 
    sess = tf.Session(config=config) 
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    weights_path = "./Documents/Python_files/NN_saved/"
    weights_path_file = ["Q_softmax_BN/Q_weights_32-0.04.hdf5"]
    models_name = ["model_BN_test"]
    
    for k in range(1):
        model = BN_quadriview_model(name = models_name[k], weights = weights_path+weights_path_file[k])
        model.compile(loss = 'binary_crossentropy', optimizer = Adam(), metrics=['accuracy'])
        X, _ = generator.get_data()
        g = K.get_session().graph
        x = tf.placeholder(tf.float32, [None, 400, 400, 1])
        y_ = tf.placeholder(tf.float32, [None, 2])
        tf_cnnvis.tf_cnnvis.activation_visualization(g, {}, input_tensor = X)
    sess.close()
    del sess
