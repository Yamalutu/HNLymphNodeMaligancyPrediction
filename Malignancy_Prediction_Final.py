import sys

# get arguments
# ct_dir = str(sys.argv[1])
# pet_dir = str(sys.argv[2])
# struct_dir = str(sys.argv[3])
# treatment_site = str(sys.argv[4])
# PET_exist = str(sys.argv[5])  #### two options: False (None) or Have

MRN = xxxxxxx # patient MRN
ct_dir = 'C:/LNMalignancyPrediction/CT_contrast'  # CT image folder
pet_dir = 'C:/LNMalignancyPrediction/PET'  # PET image folder
struct_dir = 'C:/LNMalignancyPrediction/node_structure'  # lymph node contour folder
treatment_site = 'oropharynx' ## two options: larynx and oropharynx
PET_exist = 'False'  #### two options: False (None) or Have

import subprocess
import scipy.io
import os
import numpy as np
import ast

from Cla_Comp_Model_new import Classification_Component

#extract ROI patches and radiomics features
print('extracting rois and radiomics features starts')
proc = subprocess.Popen(['exefolder/Extract_ROIs_RadiomicsFeatures_new5.exe']+[ct_dir,pet_dir,struct_dir,PET_exist],stdout=subprocess.PIPE, stderr=subprocess.PIPE)
exit_code = proc.wait()
print('extracting rois and radiomics features is done')

## CNN model prediction
roi_patch = scipy.io.loadmat(struct_dir + '/roi_patch',squeeze_me=True,struct_as_record=False)['roi_patch']

rlen = scipy.io.loadmat(struct_dir + '/roi_number',squeeze_me=True,struct_as_record=False)['len']#len(ast.literal_eval(list_struct)) #len((list_struct))#

MRN = scipy.io.loadmat(struct_dir + '/MRN',squeeze_me=True,struct_as_record=False)['MRN']

if treatment_site == 'oropharynx':
    dir_save_checkpoint_model_CT = 'AllTrainNorm_CT_BN_DP005_ClaW1v3_LR1e5_00099_0.96395_0.97959.h5'
else:
    dir_save_checkpoint_model_CT = 'LA_AllTrainNorm_CT_BN_DP005_ClaW1v1_LR1e5_00047_0.93073_0.95089.h5'
dir_save_checkpoint_model_PET_CT = 'AllTrainNorm_CT&PET_BN_DP005_ClaW1v3_LR1e5_00331_0.94694_1.00000.h5'

p = Classification_Component()
p.build_Segmen_Classi_Model(WIDTH=64, HEIGHT=64, Zs=48, channels_in=2)

## CT model
data_test = np.zeros((1, 64, 64, 48, 2))
pred_fin_ct = np.zeros((rlen, 2))
str_fin = []
count = 0
p.model.load_weights(dir_save_checkpoint_model_CT)
if rlen>1:
    for i in range(rlen):
        data_ct = roi_patch[i].ct
        name = roi_patch[i].Name

        data_ct = data_ct[4:68, 4:68, :]
        data_test[0, :, :, :, 0] = data_ct

        x = data_test
        x_2 = x[:, :, :, :, 1]
        x_2 = np.expand_dims(x_2, axis=-1)
        x_1 = x[:, :, :, :, 0]
        x_1 = np.expand_dims(x_1, axis=-1)

        prediction = p.model.predict([x, x_1, x_2], batch_size=1, verbose=0)
        pred_fin_ct[count, :] = prediction
        str_fin.append(name)
        count = count + 1
else:
    data_ct = roi_patch.ct
    name = roi_patch.Name

    data_ct = data_ct[4:68, 4:68, :]
    data_test[0, :, :, :, 0] = data_ct

    x = data_test
    x_2 = x[:, :, :, :, 1]
    x_2 = np.expand_dims(x_2, axis=-1)
    x_1 = x[:, :, :, :, 0]
    x_1 = np.expand_dims(x_1, axis=-1)

    prediction = p.model.predict([x, x_1, x_2], batch_size=1, verbose=0)
    pred_fin_ct[count, :] = prediction
    str_fin.append(name)
## PET+CT model

if PET_exist == 'Have':
    data_test = np.zeros((1, 64, 64, 48, 2))
    pred_fin = np.zeros((rlen, 2))
    str_fin = []
    count = 0
    p.model.load_weights(dir_save_checkpoint_model_PET_CT)
    if rlen>1:
        for i in range(rlen):
            data_ct = roi_patch[i].ct
            data_pet = roi_patch[i].pet
            name = roi_patch[i].Name

            data_ct = data_ct[4:68, 4:68, :]
            data_pet = data_pet[4:68, 4:68, :]
            data_pet = data_pet

            data_test[0, :, :, :, 0] = data_ct
            data_test[0, :, :, :, 1] = data_pet

            x = data_test
            x_2 = x[:, :, :, :, 1]
            x_2 = np.expand_dims(x_2, axis=-1)
            x_1 = x[:, :, :, :, 0]
            x_1 = np.expand_dims(x_1, axis=-1)

            prediction = p.model.predict([x, x_1, x_2], batch_size=1, verbose=0)
            pred_fin[count, :] = prediction
            str_fin.append(name)
            count = count + 1
    else:
        data_ct = roi_patch.ct
        data_pet = roi_patch.pet
        name = roi_patch.Name

        data_ct = data_ct[4:68, 4:68, :]
        data_pet = data_pet[4:68, 4:68, :]
        data_pet = data_pet

        data_test[0, :, :, :, 0] = data_ct
        data_test[0, :, :, :, 1] = data_pet

        x = data_test
        x_2 = x[:, :, :, :, 1]
        x_2 = np.expand_dims(x_2, axis=-1)
        x_1 = x[:, :, :, :, 0]
        x_1 = np.expand_dims(x_1, axis=-1)

        prediction = p.model.predict([x, x_1, x_2], batch_size=1, verbose=0)
        pred_fin[count, :] = prediction
        str_fin.append(name)

np.save(struct_dir + '/pred_fin_ct', pred_fin_ct)
if PET_exist == 'Have':
    np.save(struct_dir + '/pred_fin', pred_fin)

print('CNN prediction is done')
## radiomics model prediction

radi_prob = subprocess.Popen(['exefolder/radiomics_prediction.exe']+[treatment_site,struct_dir,PET_exist],stdout=subprocess.PIPE, stderr=subprocess.PIPE)
radi_prob.wait()
print('radiomics prediction is done')

## fuse model
fin_proc = subprocess.Popen(['exefolder/ER_Fusion.exe']+ [treatment_site, struct_dir,PET_exist],stdout=subprocess.PIPE, stderr=subprocess.PIPE)
fin_proc.wait()

# os.remove(struct_dir + '/fea.mat')
# os.remove(struct_dir + '/MRN.mat')
# os.remove(struct_dir + '/radi_ct_prob.mat')
# os.remove(struct_dir + '/roi_patch.mat')
# if PET_exist == 'Have':
#     os.remove(struct_dir + '/radi_pet_prob.mat')
#     os.remove(struct_dir + '/pred_fin.npy')
# os.remove(struct_dir + '/pred_fin_ct.npy')
# os.remove(struct_dir + '/roi_number.mat')
print('Finish!')






