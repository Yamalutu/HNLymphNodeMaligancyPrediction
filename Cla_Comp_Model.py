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

    def binary_PTA(self, y_true, y_pred, threshold=K.variable(value=0.5)):
        y_pred = K.cast(y_pred[:, 1] >= threshold, 'float32')
        # P = total number of positive labels
        P = K.sum(y_true[:, 1])
        # TP = total number of correct alerts, alerts from the positive class labels
        TP = K.sum(y_pred * y_true[:, 1])
        return TP / (P + K.epsilon())

    def binary_PFA(self, y_true, y_pred, threshold=K.variable(value=0.5)):
        y_pred = K.cast(y_pred[:, 0] >= threshold, 'float32')
        # N = total number of negative labels
        N = K.sum(y_true[:, 0])
        # FP = total number of false alerts, alerts from the negative class labels
        FP = K.sum(y_true[:, 0] - y_pred * y_true[:, 0])
        return FP / (N + K.epsilon())

    def auc(self, y_true, y_pred):
        ptas = tf.stack([self.binary_PTA(y_true, y_pred, k) for k in np.linspace(0, 1, 10)], axis=0)
        pfas = tf.stack([self.binary_PFA(y_true, y_pred, k) for k in np.linspace(0, 1, 10)], axis=0)
        pfas = tf.concat([tf.ones((1,)), pfas], axis=0)
        binSizes = -(pfas[1:] - pfas[:-1])
        s = ptas * binSizes
        return K.sum(s, axis=0)

    def descending_block(self, x, filter_num):
        x = BatchNormalization(axis=4)(Conv3D(filter_num, (3, 3, 3), padding='same', activation='relu')(x))
        x = BatchNormalization(axis=4)(Conv3D(filter_num, (3, 3, 3), padding='same', activation='relu')(x))
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)
        return x

    def center_block(self, x, filter_num):
        x = BatchNormalization(axis=4)(Conv2D(filter_num, (3, 3), padding='same', activation='relu')(x))
        x = BatchNormalization(axis=4)(Conv2D(filter_num, (3, 3), padding='same', activation='relu')(x))
        return x

    def ascending_block(self, x, y, filter_num):
        x = BatchNormalization(axis=4)(
            Conv2DTranspose(filter_num, (2, 2), strides=(2, 2), padding='same', activation='relu')(x))
        x = concatenate([x, y], axis=3)
        x = BatchNormalization(axis=4)(Conv2D(filter_num, (3, 3), padding='same', activation='relu')(x))
        x = BatchNormalization(axis=4)(Conv2D(filter_num, (3, 3), padding='same', activation='relu')(x))
        return x

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
################### previous architecture
#        c0 = BatchNormalization(axis=4,name='batch_normalization_c_1')(Conv3D(6, (3, 3, 3), padding='same', activation='relu', name='conv_c_1')(input))
#        c1 = BatchNormalization(axis=4,name='batch_normalization_c_2')(Conv3D(6, (3, 3, 3), padding='same', activation='relu', name='conv_c_2')(c0))
#        c2 = MaxPooling3D(pool_size=(2, 2, 2))(c1)
#
#        c3 = BatchNormalization(axis=4,name='batch_normalization_c_3')(Conv3D(12, (3, 3, 3), padding='same', activation='relu', name='conv_c_3')(c2))
#        c4 = BatchNormalization(axis=4,name='batch_normalization_c_4')(Conv3D(12, (3, 3, 3), padding='same', activation='relu', name='conv_c_4')(c3))
#        c5 = MaxPooling3D(pool_size=(2, 2, 2))(c4)
#
#        c6 = BatchNormalization(axis=4,name='batch_normalization_c_5')(Conv3D(24, (3, 3, 3), padding='same', activation='relu', name='conv_c_5')(c5))
#        c7 = BatchNormalization(axis=4,name='batch_normalization_c_6')(Conv3D(24, (3, 3, 3), padding='same', activation='relu', name='conv_c_6')(c6))
#        c8 = MaxPooling3D(pool_size=(2, 2, 2))(c7)
#
#        c9 = BatchNormalization(axis=4,name='batch_normalization_c_7')(Conv3D(48, (3, 3, 3), padding='same', activation='relu', name='conv_c_7')(c8))
#        c10 = BatchNormalization(axis=4,name='batch_normalization_c_8')(Conv3D(48, (3, 3, 3), padding='same', activation='relu', name='conv_c_8')(c9))
#        c11 = MaxPooling3D(pool_size=(2, 2, 2))(c10)
#
#        c12 = BatchNormalization(axis=4,name='batch_normalization_c_9')(Conv3D(96, (3, 3, 3), padding='same', activation='relu', name='conv_c_9')(c11))
#        c13 = BatchNormalization(axis=4,name='batch_normalization_c_10')(Conv3D(96, (3, 3, 3), padding='same', activation='relu', name='conv_c_10')(c12))
#        c14 = MaxPooling3D(pool_size=(2, 2, 2))(c13)
#
#        s6 = Flatten(name='flatten')(c14)#GlobalAveragePooling3D(name='GP_3D')(c14)#
#        s7 = Dense(100, activation='relu',name='fc_1')(s6)
#        s8 = Dense(10, activation='relu',name='fc_2')(s7)
#        s11 = Dense(2, activation='softmax', name='prediction')(s8)
####################### end
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

        #self.model.summary()

    def load_train_data_batch_generator(self, batch_size=32, rows_in=64, cols_in=64, zs_in=48, channels_in=2,
                                        channels_out=2, rescale_in=1, dir_in_train=None, dir_out_train_label=None,
                                        dir_in_train_PET=None, dir_out_train_seg=None):

        x = np.zeros((batch_size, rows_in, cols_in, zs_in, channels_in))
        y = np.zeros((batch_size, rows_in, cols_in, zs_in, 1))
        y1 = np.zeros((batch_size, channels_out))

        while True:
            count = 0
            for rfile in np.random.choice(os.listdir(dir_out_train_seg), batch_size, replace=False):
                data = scipy.io.loadmat(os.path.join(dir_in_train, rfile))['roi_patch_ct']
                data_PET = scipy.io.loadmat(os.path.join(dir_in_train_PET, rfile))['roi_patch_pet']
                seg = scipy.io.loadmat(os.path.join(dir_out_train_seg, rfile))['roi_patch_seg']
                seg[np.where(seg > 1)] = 1

                data = data[4:68, 4:68, :]
                data_PET = data_PET[4:68, 4:68, :]
                seg = seg[4:68, 4:68, :]

                label = scipy.io.loadmat(os.path.join(dir_out_train_label, rfile))['label']
                x[count, :, :, :, 0] = np.reshape(data, (rows_in, cols_in, zs_in), order='F')
                x[count, :, :, :, 1] = np.reshape(data_PET, (rows_in, cols_in, zs_in), order='F')
                y1[count, :] = np.reshape(label, (channels_out), order='F')
                count += 1

            x_2 = x[:, :, :, :, 1]
            x_2 = np.expand_dims(x_2, axis=-1)
            x_1 = x[:, :, :, :, 0]
            x_1 = np.expand_dims(x_1, axis=-1)

            self.x_train = [x, x_1, x_2]
            self.y_train = y1
            yield (self.x_train, self.y_train)

    def load_val_data_batch_generator(self, batch_size=32, rows_in=64, cols_in=64, zs_in=48, channels_in=2,
                                      channels_out=2, rescale_in=1, dir_in_val=None, dir_out_val_label=None,
                                      dir_in_val_PET=None, dir_out_val_seg=None):

        x = np.zeros((batch_size, rows_in, cols_in, zs_in, channels_in))
        y = np.zeros((batch_size, rows_in, cols_in, zs_in, 1))
        y1 = np.zeros((batch_size, channels_out))

        while True:
            count = 0
            for rfile in np.random.choice(os.listdir(dir_out_val_seg), batch_size, replace=False):
                data = scipy.io.loadmat(os.path.join(dir_in_val, rfile))['roi_patch_ct']
                data_PET = scipy.io.loadmat(os.path.join(dir_in_val_PET, rfile))['roi_patch_pet']
                seg = scipy.io.loadmat(os.path.join(dir_out_val_seg, rfile))['roi_patch_seg']
                seg[np.where(seg > 1)] = 1

                data = data[4:68, 4:68, :]
                data_PET = data_PET[4:68, 4:68, :]
                seg = seg[4:68, 4:68, :]

                label = scipy.io.loadmat(os.path.join(dir_out_val_label, rfile))['label']
                x[count, :, :, :, 0] = np.reshape(data, (rows_in, cols_in, zs_in), order='F')
                x[count, :, :, :, 1] = np.reshape(data_PET, (rows_in, cols_in, zs_in), order='F')
                y1[count, :] = np.reshape(label, (channels_out), order='F')
                count += 1

            x_2 = x[:, :, :, :, 1]
            x_2 = np.expand_dims(x_2, axis=-1)
            x_1 = x[:, :, :, :, 0]
            x_1 = np.expand_dims(x_1, axis=-1)

            self.x_val = [x, x_1, x_2]
            self.y_val = y1
            yield (self.x_val, self.y_val)

    def train_LN_category_pred_batch_add_history(self, batch_size=32, batch_size_val=10, epochs=5, verbose=1,
                                                 dir_in_train=None,
                                                 dir_in_train_PET=None, dir_out_train_label=None,
                                                 dir_out_train_seg=None, dir_in_val=None, dir_in_val_PET=None,
                                                 dir_out_val_label=None, dir_out_val_seg=None, rescale_in=1,
                                                 dir_save_checkpoint_model=None):

        rows_in = 64
        cols_in = 64
        zs_in = 48
        channels_in = 2
        channels_out = 2

        model_checkpoint = ModelCheckpoint(
            os.path.join(dir_save_checkpoint_model,
                         'ClaModule_CT&PET_BN_DP005_ClaW1v5_LR1e5_{epoch:05d}_{sensitivity:.5f}_{val_sensitivity:.5f}.hdf5'),
            monitor='val_M', save_best_only=True, save_weights_only=True, mode='max')

        self.model.fit_generator(generator=self.load_train_data_batch_generator(batch_size=batch_size,
                                                                                rows_in=rows_in,
                                                                                cols_in=cols_in,
                                                                                zs_in=zs_in,
                                                                                channels_in=channels_in,
                                                                                channels_out=channels_out,
                                                                                rescale_in=rescale_in,
                                                                                dir_in_train=dir_in_train,
                                                                                dir_out_train_label=dir_out_train_label,
                                                                                dir_in_train_PET=dir_in_train_PET,
                                                                                dir_out_train_seg=dir_out_train_seg),
                                 steps_per_epoch=round(len(os.listdir(dir_out_train_seg)) / batch_size),
                                 class_weight={0: 1, 1: 2},
                                 epochs=epochs,
                                 verbose=verbose,
                                 callbacks=[model_checkpoint],
                                 validation_data=self.load_val_data_batch_generator(batch_size=batch_size_val,
                                                                                    rows_in=rows_in,
                                                                                    cols_in=cols_in,
                                                                                    zs_in=zs_in,
                                                                                    channels_in=channels_in,
                                                                                    channels_out=channels_out,
                                                                                    rescale_in=rescale_in,
                                                                                    dir_in_val=dir_in_val,
                                                                                    dir_out_val_label=dir_out_val_label,
                                                                                    dir_in_val_PET=dir_in_val_PET,
                                                                                    dir_out_val_seg=dir_out_val_seg),
                                 validation_steps=round(len(os.listdir(dir_out_val_seg)) / batch_size_val))
