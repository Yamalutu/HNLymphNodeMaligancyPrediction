# HNLymphNodeMaligancyPrediction

##### To utilize this prediction package, please install the following applications in advance:

(1). matlab runtime 9.2
(2). python 3.6
(3). tensorflow 1.12

##### To get the malignancy prediction of the HN lymph nodes, please run Malignancy_Prediction_Final.py

##### Inside of Malignancy_Prediction_Final.py, please update the folder directories in which CT images, PET images, and lymph node contours exsit respectively as well as the patient MRN based on your own test data. Especially, in the node_structure folder, there are two items: one is the exported structure set which needs a specific name as "AIR.dcm", and the other item is a text file with a specific name as "Structures.txt". In the "Structures.txt", you just need to list the lymph node names that you want to do the prediction.

##### In the "exefolder", radiomics feature extraction, radiomics prediction and ER fusion exes are available.

##### Trained Models are available in the following link:
https://drive.google.com/drive/folders/14pWvh-vqjaFe9cLS7usEwDg9svJR0foH?usp=share_link

##### To get more details of the prediction algorithms, please refer to the following paper:

1. Combining many-objective radiomics and 3D convolutional neural network through evidential reasoning to predict lymph node metastasis in head and neck cancer
