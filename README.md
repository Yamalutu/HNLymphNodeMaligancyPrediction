# HNLymphNodeMaligancyPrediction

##### To utilize this prediction package, please install the following applications in advance:

(1) matlab runtime 9.2 (installation time: 1 hour)

(2) python 3.6 (installation time: 10 mins if installing python 3.6 from anaconda environment) 

(3) tensorflow 1.12 (installation time: 1 hour)

installing tensorflow 1.12 through anaconda is recommended by using following commands:

conda create -n tensorflow==1.12.0

conda activate tensorflow==1.12.0

##### Matlab exes including roi_feature_extraction.exe, radiomics_prediction.exe and ER_fusion.exe are compressed into two RAR files: matlab exe.part1.rar and matlab exe.part2.rar

##### Pre-trained CNN models are compressed into five RAR files: trained CNN models.part1.rar, trained CNN models.part2.rar, trained CNN models.part3.rar, trained CNN models.part4.rar, and trained CNN models.part5.rar.

##### An anonymized example case including CT images, PET images, lymph node strucgtures and the expected predition results for each lymph node are available to download in the following link:

https://drive.google.com/drive/folders/1cMNPtuoSCYgQKU0AYcSepu5llM9F5cRo?usp=share_link     

##### To get the malignancy prediction of the HN lymph nodes, please run Malignancy_Prediction_Final.py

##### Inside of Malignancy_Prediction_Final.py, please update the folder directories in which CT images, PET images, and lymph node contours exsit respectively as well as the patient MRN based on your own test data. Especially, in the node_structure folder, there are two items: one is the exported structure set which needs a specific name as "AIR.dcm", and the other item is a text file with a specific name as "Structures.txt". In the "Structures.txt", you just need to list the lymph node names that you want to do the prediction. 

##### This package was used to generate lymph node malignancy prediction results for a manuscript submitted to Nature Medicine.

