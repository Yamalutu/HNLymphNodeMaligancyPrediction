# HNLymphNodeMaligancyPrediction

##### To utilize this prediction package, please install the following applications in advance:

(1). matlab runtime 9.2
(2). python 3.6
(3). tensorflow 1.12 

installing tensorflow 1.12 through anaconda is recommended by using following commands:

conda create -n tensorflow==1.12.0
conda activate tensorflow==1.12.0

##### To get the malignancy prediction of the HN lymph nodes, please run Malignancy_Prediction_Final.py

##### Inside of Malignancy_Prediction_Final.py, please update the folder directories in which CT images, PET images, and lymph node contours exsit respectively as well as the patient MRN based on your own test data. Especially, in the node_structure folder, there are two items: one is the exported structure set which needs a specific name as "AIR.dcm", and the other item is a text file with a specific name as "Structures.txt". In the "Structures.txt", you just need to list the lymph node names that you want to do the prediction. 

##### an anonymized dataset for testing has been provided in the example folder

##### Three trained CNN models and radiomics_prediction.exe are available in the following link:
https://drive.google.com/drive/folders/14pWvh-vqjaFe9cLS7usEwDg9svJR0foH?usp=share_link

##### This package was used to generate lymph node malignancy prediction results for a manuscript submitted to Nature Medicine.

