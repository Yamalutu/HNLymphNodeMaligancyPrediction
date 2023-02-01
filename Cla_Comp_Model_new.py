import h5py
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import BatchNormalization, Activation, Conv2DTranspose, concatenate, Flatten, Dense, \
    GlobalAveragePooling3D, Multiply, Conv3DTranspose, Dropout
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Conv3D, MaxPooling3D
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Reshape, dot, add, Conv1D, Conv3D, add
from tensorflow.python.keras.layers import MaxPool1D
#from tensorflow.python.keras.losses import binary_crossentropy

from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
import math
from tensorflow.python.keras.callbacks import ModelCheckpoint
import scipy.io
import os
import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.keras import backend as KC
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib.keras.api.keras.regularizers import l2


class Classification_Component(object):
    
    def conv3d_bn(self,x,
                  filters,
                  num_row,
                  num_col,
                  num_slice,
                  padding='same',
                  strides=(1, 1, 1),
                  name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
        if K.image_data_format() == 'channels_first':
            bn_axis = 1
        else:
            bn_axis = 4
        x = Conv3D(
            filters, (num_row, num_col, num_slice),
            strides=strides,
            padding=padding,
            use_bias=False,
            name=conv_name)(x)
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        x = Activation('relu', name=name)(x)
        x = Dropout(0.05)(x)  #0.1
        #x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        return x

    def sensitivity(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true[:, 1] * y_pred[:, 1], 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true[:, 1], 0, 1)))
        return true_positives / (possible_positives + K.epsilon())

    def specificity(self, y_true, y_pred):
        true_negatives = K.sum(K.round(K.clip(y_true[:, 0] * y_pred[:, 0], 0, 1)))
        possible_negatives = K.sum(K.round(K.clip(y_true[:, 0], 0, 1)))
        return true_negatives / (possible_negatives + K.epsilon())
        
    def M(self, y_true, y_pred):

        sen = self.sensitivity(y_true, y_pred)
        spe = self.specificity(y_true, y_pred)

        T = K.variable(value=1.5) * sen + spe

        threshold = K.variable(value=0.80)
        rate1 = K.cast(sen >= threshold, 'float32')
        rate2 = K.cast(spe >= threshold, 'float32')
        T = T * rate1 * rate2
        return T      

    def load_model(self, load_file_path, load_file_name):
        self.model.save(os.path.join(load_file_path, load_file_name))

    def save_model(self, save_file_path, save_file_name):
        self.model.save(os.path.join(save_file_path, save_file_name))

    def load_weights(self, load_file_path, load_file_name):
        self.model.load_weights(os.path.join(load_file_path, load_file_name))

    def build_Segmen_Classi_Model(self, WIDTH=72, HEIGHT=72, Zs=48, channels_in=2):

        input = Input(shape=(WIDTH, HEIGHT, Zs, channels_in))
        ct_tensor = Input((WIDTH, HEIGHT, Zs, 1))
        pet_tensor = Input((WIDTH, HEIGHT, Zs, 1))

        s1 = self.conv3d_bn(input, 64, 5, 5, 5, strides=(1, 1, 1), padding='valid')
        s2 = self.conv3d_bn(s1, 64, 3, 3, 3, strides=(1, 1, 1), padding='valid')
        s3 = self.conv3d_bn(s2, 64, 3, 3, 3, strides=(1, 1, 1), padding='valid')
        s4 = self.conv3d_bn(s3, 64, 3, 3, 3, strides=(1, 1, 1), padding='valid')
        s5 = self.conv3d_bn(s4, 64, 3, 3, 3, strides=(1, 1, 1), padding='valid')
        s6 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(s5)
        s7 = self.conv3d_bn(s6, 64, 3, 3, 3, padding='valid')
        s8 = self.conv3d_bn(s7, 64, 2, 2, 2, padding='valid')
        s9 = self.conv3d_bn(s8, 64, 3, 3, 3)
        s10 = MaxPooling3D((3, 3, 3), strides=(2, 2, 2))(s9)
        s11 = self.conv3d_bn(s10, 64, 3, 3, 3, padding='same')
        s12 = self.conv3d_bn(s11, 64, 3, 3, 1, padding='same')
        s13 = self.conv3d_bn(s12, 64, 3, 3, 1, padding='same')
        s14 = self.conv3d_bn(s13, 32, 3, 3, 1, padding='same')
        s15 = Flatten()(s14)
        s16 = Dense(256)(s15)
        s17 = Dense(2, activation='softmax', name='predictions')(s16)

        self.model = Model(
            inputs=[input, ct_tensor, pet_tensor],
            outputs=[s17])

        self.model.compile(
            loss=['binary_crossentropy'], metrics=['acc',self.sensitivity, self.M],
            optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0))






