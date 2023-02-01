import h5py
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2DTranspose, concatenate, Flatten, Dense, \
    GlobalAveragePooling3D, Multiply, Conv3DTranspose, Dropout, GlobalMaxPooling3D, GlobalAveragePooling3D, \
    LayerNormalization
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Lambda, Reshape, dot, add, Conv1D, Conv3D, add, \
    MaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import math
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import Callback
import scipy.io
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import backend as KC
from tensorflow.keras.regularizers import l2
from scipy.ndimage import zoom
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix, roc_auc_score
from tensorflow.keras.metrics import AUC, SensitivityAtSpecificity, SpecificityAtSensitivity


class Segmentation_Classification_SIM(object):
    def conv3d_bn(self, x,
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
            use_bias=True,
            name=conv_name)(x)
        # x = BatchNormalization(axis=bn_axis, scale=True, name=bn_name)(x)  #LayerNormalization(axis=-1 , center=True , scale=True)(x)#
        x = Activation('relu', name=name)(x)
        # x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        x = Dropout(0.01)(x)
        # x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        return x

    def sensitivity(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true[:, 1] * y_pred[:, 1], 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true[:, 1], 0, 1)))
        return true_positives / (possible_positives + K.epsilon())

    def specificity(self, y_true, y_pred):
        true_negatives = K.sum(K.round(K.clip(y_true[:, 0] * y_pred[:, 0], 0, 1)))
        possible_negatives = K.sum(K.round(K.clip(y_true[:, 0], 0, 1)))
        return true_negatives / (possible_negatives + K.epsilon())

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

    def weighted_acc(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true[:, 1] * y_pred[:, 1], 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true[:, 1], 0, 1)))
        true_negatives = K.sum(K.round(K.clip(y_true[:, 0] * y_pred[:, 0], 0, 1)))
        possible_negatives = K.sum(K.round(K.clip(y_true[:, 0], 0, 1)))
        return (4 * true_positives + true_negatives) / (4 * possible_positives + possible_negatives)

    def weighted_bce(self, y_true, y_pred):
        a = K.sum(y_true[:, 0])
        b = K.sum(y_pred[:, 1])
        w_a = b / (a + b) * y_true[:, 0]
        w_b = a / (a + b) * y_true[:, 1]
        w = w_a + w_b
        return -w * tf.reduce_sum(y_true * tf.math.log(y_pred), axis=len(y_true.get_shape()) - 1)

    def dice_coef(self, y_true, y_pred, smooth=1):

        y_pred = K.clip(y_pred, 0, 1)
        intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
        union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
        return abs(K.mean((2. * intersection + smooth) / (union + smooth), axis=0))

    def dice_loss(self, y_true, y_pred):
        return 1 - self.dice_coef(y_true, y_pred)

    def dice_coef_new(self, y_true, y_pred, smooth=1):

        y_pred = K.clip(y_pred, 0, 1)
        y_pred = K.reshape(y_pred[:, :, :, :, 0], (24, 64, 64, 48, 1))
        intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
        union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
        return abs(K.mean((2. * intersection + smooth) / (union + smooth), axis=0))

    def dice_loss_new(self, y_true, y_pred):
        return 1 - self.dice_coef_new(y_true, y_pred)

    def dice_sparse_loss_new(self, y_true, y_pred):
        th = K.variable(K.constant(0.5))
        y_pred = y_pred / tf.reduce_max(y_pred)
        y_pred_1 = K.cast(y_pred * y_true >= th, 'float32')
        y_pred_2 = K.cast(y_pred * (1 - y_true) >= th, 'float32')
        p1 = 1 - self.dice_coef(y_true, y_pred_1 * y_pred)
        p2 = K.sum(abs(y_pred_2)) / K.sum(1 - y_true)
        return p1 + 0.01 * p2

    def dice_sparse_loss(self, y_true, y_pred):
        th = K.variable(K.constant(0.5))
        y_pred_1 = K.cast(y_pred * y_true >= th, 'float32')
        p1 = 1 - self.dice_coef(y_true, y_pred)
        p2 = K.sum(abs(y_pred * (1 - y_true))) / K.sum(y_true)
        return p1 + p2

    def proposed_loss(self, y_true, y_pred):
        th_ind = K.variable(K.constant(0.5))
        th_out = K.variable(K.constant(0.5))
        y_pred_ind = K.cast(y_pred >= th_ind, 'float32') * y_true
        y_pred_out = K.cast(y_pred >= th_out, 'float32') * (1 - y_true)
        lambda_out = K.variable(K.constant(0.5))
        return 1 - self.dice_coef(y_true, y_pred_ind) + lambda_out * K.mean(y_pred_out)

    def auc_dice(self, y_true, y_pred):
        dic = self.dice_coef(y_true, y_pred, smooth=1)
        auc = self.auc(y_true, y_pred)
        return dic + auc

    def modified_dice_loss(self, y_true, y_pred):
        y_pred_new = self.binary_activation(y_pred)
        return 1 - self.dice_coef(y_true, y_pred_new)

    def modified_dice_l1(self, y_true, y_pred):
        y_pred_new = self.binary_activation(y_pred)
        a = self.dice_coef(y_true, y_pred_new)
        b = abs(y_pred_new * (1 - y_true))
        b1 = K.sum(b) / K.sum(1 - y_true)
        return K.mean(a) + K.mean(b1)

    def dual_cross_entropy(self, y_true, y_pred):
        return -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=len(y_true.get_shape()) - 1) + 4.5 * tf.reduce_sum(
            (1 - y_true) * tf.math.log(1 + y_pred), axis=len(y_true.get_shape()) - 1)  ##previously beta = 2

    def proposed_foco_dual_loss(self, y_true, y_pred):
        gamma = 2  # 2.0
        alpha = 0.25  # 0.5
        beta = 1 / 2  # 0.5
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        focal_loss = -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + 1e-3)) - K.mean(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + 1e-3))
        dual_loss_reg = beta * tf.reduce_sum((1 - y_true) * tf.math.log(1 + y_pred), axis=len(y_true.get_shape()) - 1)
        return focal_loss + dual_loss_reg

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

    def seg_loss(self, y_true, y_pred):
        a = abs(y_true - y_pred) * y_true
        return K.sum(a) / K.sum(y_true)

    def seg_sparse_loss(self, y_true, y_pred):
        a = abs(y_true - y_pred) * y_true
        b = abs(y_pred * (1 - y_true))
        c = K.sum(1 - y_true) / K.sum(y_true) * a + b
        return K.mean(c)

    def binary_activation(self, tensor):
        threshold = 0.5
        cond = tf.less(tensor, threshold * tf.ones(tf.shape(tensor)))
        out = tf.where(cond, tf.zeros(tf.shape(tensor)), tf.ones(tf.shape(tensor)))
        return out

    def threshold_seg(self, args):
        ct, seg = args[0], args[1]
        cond = tf.equal(ct, 0 * tf.ones(tf.shape(ct)))
        out = tf.where(cond, tf.zeros(tf.shape(ct)), seg)
        return out

    def seg_sparse_threshold_loss(self, y_true, y_pred):
        b = abs(y_pred * (1 - y_true))
        y_pred_new = self.binary_activation(y_pred)
        a = abs(y_true - y_pred_new) * y_true
        a1 = K.sum(a) / K.sum(y_true)
        b1 = K.sum(b) / K.sum(1 - y_true)
        c = a1 + 0.7 * b1
        return c

    def M(self, y_true, y_pred):

        sen = self.sensitivity(y_true, y_pred)
        spe = self.specificity(y_true, y_pred)

        T = 1.2 * sen + spe

        threshold = 0.6
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

    def layer_sum(self, args):
        fea1, fea2 = args[0],args[1]
        return 0.5*fea1+0.5*fea2

    def build_Segmen_Classi_Model(self, WIDTH=72, HEIGHT=72, Zs=48, channels_in=2):

        input = Input(shape=(WIDTH, HEIGHT, Zs, channels_in),name='input_tumor')
        input2 = Input(shape=(WIDTH, HEIGHT, Zs, channels_in),name='input_roi')


        s1 = self.conv3d_bn(input, 8, 3, 3, 3, padding='same')
        s2 = self.conv3d_bn(s1, 8, 3, 3, 3, padding='same')
        s3 = self.conv3d_bn(s2, 8, 3, 3, 3, padding='same')
        s5 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(s3)
        s6 = self.conv3d_bn(s5, 16, 3, 3, 3, padding='same')
        s7 = self.conv3d_bn(s6, 16, 3, 3, 3, padding='same')
        s8 = self.conv3d_bn(s7, 16, 3, 3, 3, padding='same')
        s10 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(s8)
        s11 = self.conv3d_bn(s10, 32, 3, 3, 3, padding='same')
        s12 = self.conv3d_bn(s11, 32, 3, 3, 3, padding='same')
        s13 = self.conv3d_bn(s12, 32, 3, 3, 3, padding='same')
        s14 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(s13)
        s15 = self.conv3d_bn(s14, 64, 3, 3, 3, padding='same')
        s16 = self.conv3d_bn(s15, 64, 3, 3, 3, padding='same')
        s17 = self.conv3d_bn(s16, 64, 3, 3, 3, padding='same')

        s18 = GlobalAveragePooling3D()(s17)
        s20 = Dense(2, kernel_regularizer=l2(0.01), activation='softmax', name='predictions_tumor')(s18)

        c1 = self.conv3d_bn(input2, 8, 3, 3, 3, padding='same')
        c2 = self.conv3d_bn(c1, 8, 3, 3, 3, padding='same')
        c3 = self.conv3d_bn(c2, 8, 3, 3, 3, padding='same')
        c5 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(c3)
        c6 = self.conv3d_bn(c5, 16, 3, 3, 3, padding='same')
        c7 = self.conv3d_bn(c6, 16, 3, 3, 3, padding='same')
        c8 = self.conv3d_bn(c7, 16, 3, 3, 3, padding='same')
        c10 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(c8)
        c11 = self.conv3d_bn(c10, 32, 3, 3, 3, padding='same')
        c12 = self.conv3d_bn(c11, 32, 3, 3, 3, padding='same')
        c13 = self.conv3d_bn(c12, 32, 3, 3, 3, padding='same')
        c14 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(c13)
        c15 = self.conv3d_bn(c14, 64, 3, 3, 3, padding='same')
        c16 = self.conv3d_bn(c15, 64, 3, 3, 3, padding='same')
        c17 = self.conv3d_bn(c16, 64, 3, 3, 3, padding='same')

        c18 = GlobalAveragePooling3D()(c17)
        c20 = Dense(2, kernel_regularizer=l2(0.01), activation='softmax', name='predictions_roi')(c18)

# ###################### Arc1 ##########################################
#         pred = Lambda(self.layer_sum, name='predictions')([s20,c20])

###################### Arc2 ##########################################
        fea_comb = concatenate([s18,c18], axis=1)
        fea_comb = Dense(64,kernel_regularizer= l2(0.01), activation='relu')(fea_comb)
        pred = Dense(2, kernel_regularizer=l2(0.01), activation='softmax')(fea_comb)

        self.model = Model(
            inputs=[input, input2],
            outputs=pred)

        self.model.compile(
            loss=self.proposed_foco_dual_loss,#'binary_crossentropy',  # self.dual_cross_entropy,#self.weighted_bce,#self.proposed_foco_dual_loss,#
            metrics=['acc', AUC(num_thresholds=10, curve='PR', summation_method='minoring'),
                     SensitivityAtSpecificity(0.85, num_thresholds=10),
                     SpecificityAtSensitivity(0.7, num_thresholds=10)],  # dice_loss
            optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0))

        self.model.summary()

    def load_train_data_batch_generator(self, batch_size=32, rows_in=64, cols_in=64, zs_in=48, channels_in=2,
                                        channels_out=2, rescale_in=1, dir_in_train=None, dir_out_train_label=None,
                                        dir_in_train_PET=None, dir_out_train_seg=None, dir_out_train_mask=None):

        x = np.zeros((batch_size + batch_size, rows_in, cols_in, zs_in, channels_in))
        x1 = np.zeros((batch_size + batch_size, rows_in, cols_in, zs_in, channels_in))
        y1 = np.zeros((batch_size + batch_size, channels_out))

        lst = []
        lst_label = []
        lst_dir = []
        lst_pet_dir = []
        lst_seg_dir = []
        lst_label_dir = []
        lst_mask_dir = []
        ## surpose dir_in_train contains six different folders
        for i in range(len(dir_in_train)):
            for j in range(len(os.listdir(dir_in_train[i]))):
                lst.append(os.listdir(dir_in_train[i])[j])
                lst_label.append(
                    scipy.io.loadmat(os.path.join(dir_out_train_label[i], os.listdir(dir_in_train[i])[j]))['label'][0][
                        1])
                lst_dir.append(dir_in_train[i])
                lst_pet_dir.append(dir_in_train_PET[i])
                lst_seg_dir.append(dir_out_train_seg[i])
                lst_mask_dir.append(dir_out_train_mask[i])
                lst_label_dir.append(dir_out_train_label[i])
        lst_label_array = np.array(lst_label)
        lst_label_pos = np.where(lst_label_array == 1)
        lst_label_neg = np.where(lst_label_array == 0)

        while True:
            count = 0
            for rfile in np.random.choice(lst_label_pos[0], batch_size, replace=False):
                data = \
                    scipy.io.loadmat(os.path.join(lst_dir[rfile], lst[rfile]), verify_compressed_data_integrity=False)[
                        'roi_patch_ct']
                data_PET = \
                    scipy.io.loadmat(os.path.join(lst_pet_dir[rfile], lst[rfile]),
                                     verify_compressed_data_integrity=False)[
                        'roi_patch_pet']
                seg = \
                    scipy.io.loadmat(os.path.join(lst_seg_dir[rfile], lst[rfile]),
                                     verify_compressed_data_integrity=False)[
                        'roi_patch_seg']
                mask = \
                    scipy.io.loadmat(os.path.join(lst_mask_dir[rfile], lst[rfile]),
                                     verify_compressed_data_integrity=False)[
                        'roi_patch_seg']

                seg = seg[:, :, 32:96]
                data = data[:, :, 32:96]
                data_PET = data_PET[:, :, 32:96]
                mask = mask[:, :, 32:96]

                seg = seg[::2, ::2, ::2]
                data = data[::2, ::2, ::2]
                data_PET = data_PET[::2, ::2, ::2]
                mask = mask[::2, ::2, ::2]

                data_PET = data_PET / 11

                label = scipy.io.loadmat(os.path.join(lst_label_dir[rfile], lst[rfile]))['label']
                x[count, :, :, :, 0] = np.reshape(data * seg, (rows_in, cols_in, zs_in), order='F')
                x[count, :, :, :, 1] = np.reshape(data_PET * seg, (rows_in, cols_in, zs_in), order='F')
                x1[count, :, :, :, 0] = np.reshape(data * mask, (rows_in, cols_in, zs_in), order='F')
                x1[count, :, :, :, 1] = np.reshape(data_PET * mask, (rows_in, cols_in, zs_in), order='F')
                y1[count, :] = np.reshape(label, (channels_out), order='F')
                count += 1
            for rfile in np.random.choice(lst_label_neg[0], batch_size, replace=False):
                data = \
                    scipy.io.loadmat(os.path.join(lst_dir[rfile], lst[rfile]), verify_compressed_data_integrity=False)[
                        'roi_patch_ct']
                data_PET = \
                    scipy.io.loadmat(os.path.join(lst_pet_dir[rfile], lst[rfile]),
                                     verify_compressed_data_integrity=False)[
                        'roi_patch_pet']
                seg = \
                    scipy.io.loadmat(os.path.join(lst_seg_dir[rfile], lst[rfile]),
                                     verify_compressed_data_integrity=False)[
                        'roi_patch_seg']

                mask = \
                    scipy.io.loadmat(os.path.join(lst_mask_dir[rfile], lst[rfile]),
                                     verify_compressed_data_integrity=False)[
                        'roi_patch_seg']

                seg = seg[:, :, 32:96]
                data = data[:, :, 32:96]
                data_PET = data_PET[:, :, 32:96]
                mask = mask[:, :, 32:96]

                seg = seg[::2, ::2, ::2]
                data = data[::2, ::2, ::2]
                data_PET = data_PET[::2, ::2, ::2]
                mask = mask[::2, ::2, ::2]

                data_PET = data_PET / 11

                label = scipy.io.loadmat(os.path.join(lst_label_dir[rfile], lst[rfile]))['label']
                x[count, :, :, :, 0] = np.reshape(data * seg, (rows_in, cols_in, zs_in), order='F')
                x[count, :, :, :, 1] = np.reshape(data_PET * seg, (rows_in, cols_in, zs_in), order='F')
                x1[count, :, :, :, 0] = np.reshape(data * mask, (rows_in, cols_in, zs_in), order='F')
                x1[count, :, :, :, 1] = np.reshape(data_PET * mask, (rows_in, cols_in, zs_in), order='F')
                y1[count, :] = np.reshape(label, (channels_out), order='F')
                count += 1

            self.x_train = {'input_tumor': x, 'input_roi': x1}
            self.y_train = y1
            yield (self.x_train, self.y_train)

    def train_LN_category_pred_batch_add_history(self, batch_size=32, batch_size_val=10, verbose=1,
                                                 dir_in_train=None,
                                                 dir_in_train_PET=None, dir_out_train_label=None,
                                                 dir_out_train_seg=None, dir_out_train_mask=None, dir_in_val=None, dir_in_val_PET=None,
                                                 dir_out_val_label=None, dir_out_val_seg=None, dir_out_val_mask=None, rescale_in=1,
                                                 dir_save_checkpoint_model=None, Val_ind=None, Test_ind=None,
                                                 Cla_epoch=None):

        rows_in = 64
        cols_in = 64
        zs_in = 32
        channels_in = 2
        channels_out = 2
        sample_size = [63, 63, 63, 63, 64]

        for j in range(Cla_epoch):
            self.model.fit(self.load_train_data_batch_generator(batch_size=batch_size, rows_in=rows_in,
                                                                cols_in=cols_in,
                                                                zs_in=zs_in,
                                                                channels_in=channels_in,
                                                                channels_out=channels_out,
                                                                rescale_in=rescale_in,
                                                                dir_in_train=dir_in_train,
                                                                dir_out_train_label=dir_out_train_label,
                                                                dir_in_train_PET=dir_in_train_PET,
                                                                dir_out_train_seg=dir_out_train_seg,
                                                                dir_out_train_mask=dir_out_train_mask), verbose=verbose,
                           steps_per_epoch=16, epochs=1)

            len_test = sample_size[Val_ind[0] - 1]
            data_test = np.zeros((1, 64, 64, 32, 2))
            pred_fin = np.zeros((len_test, 2))
            gt_fin = np.zeros((len_test, 2))
            lst = os.listdir(dir_in_val)
            lst.sort()

            count = 0
            for i in range(len_test):
                data_ct = scipy.io.loadmat(os.path.join(dir_in_val, lst[i]))['roi_patch_ct']
                data_PET = scipy.io.loadmat(os.path.join(dir_in_val_PET, lst[i]))['roi_patch_pet']
                seg = scipy.io.loadmat(os.path.join(dir_out_val_seg, lst[i]))['roi_patch_seg']
                mask = scipy.io.loadmat(os.path.join(dir_out_val_mask, lst[i]))['roi_patch_seg']

                data_ct = data_ct[:, :, 32:96]
                data_PET = data_PET[:, :, 32:96]
                seg = seg[:, :, 32:96]
                mask = mask[:, :, 32:96]

                seg = seg[::2, ::2, ::2]
                mask = mask[::2, ::2, ::2]
                data_ct = data_ct[::2, ::2, ::2]
                data_PET = data_PET[::2, ::2, ::2]

                data_test[0, :, :, :, 0] = np.reshape(data_ct * seg, (64, 64, 32), order='F')
                data_test[0, :, :, :, 1] = np.reshape(data_PET / 11 * seg, (64, 64, 32), order='F')

                data_test1[0, :, :, :, 0] = np.reshape(data_ct * mask, (64, 64, 32), order='F')
                data_test1[0, :, :, :, 1] = np.reshape(data_PET / 11 * mask, (64, 64, 32), order='F')

                pre = self.model.predict({'input_roi': data_test, 'input_tumor': data_test1}, batch_size=1, verbose=0)

                label = scipy.io.loadmat(os.path.join(dir_out_val_label, lst[i]))['label']

                pred_fin[count, :] = pre
                gt_fin[count, :] = label
                count += 1

            cf = confusion_matrix(np.argmax(gt_fin, axis=1), np.argmax(pred_fin, axis=1))
            acc = (cf[0, 0] + cf[1, 1]) / np.sum(cf)
            sen = (cf[1, 1]) / (cf[1, 0] + cf[1, 1])
            spe = (cf[0, 0]) / (cf[0, 0] + cf[0, 1])
            auc = roc_auc_score(gt_fin, pred_fin)
            prauc = average_precision_score(gt_fin, pred_fin, average='micro', pos_label=1, sample_weight=None)
            print('Confusion matrix' + ':', 'acc:', acc, 'sen:', sen, 'spe:', spe, 'auc:', auc, 'prauc:', prauc)
            self.model.save(os.path.join(dir_save_checkpoint_model, 'F' + str(Val_ind) + 'Val_F' + str(
                Test_ind) + 'Test_Com_PL2_Epoch_' + str(j) + '_VACC_' + str(round(acc, 2)) + '_VSen_' + str(
                round(sen, 2)) + '_VAUC_' + str(round(auc, 2)) + '_VMPR_' + str(round(prauc, 2)) + '.h5'))
