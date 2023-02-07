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

##### This software package generated the lymph node malignancy predictions for a manuscript submitted to Nature Medicine. The executable files were split and password protected using WinRar. To extract the .part files, use WinRar or 7-Zip (https://www.7-zip.org/), and enter the password provided during the manuscript submission process.

Copyright (c) 2003 The University of Texas Southwestern Medical Center.

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted for academic research use only (subject to the limitations in the disclaimer below) provided that the following conditions are met: 

*Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

*Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

*Neither the name of the copyright holders nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

ANY USE OR REDISTRIBUTION OF THIS SOFTWARE FOR COMMERCIAL PURPOSES, WHETHER IN SOURCE OR BINARY FORM, WITH OR WITHOUT MODIFICATION, IS EXPRESSLY PROHIBITED; ANY USE OR REDISTRIBUTION BY A FOR-PROFIT ENTITY SHALL COMPRISE USE OR REDISTRIBUTION FOR COMMERCIAL PURPOSES.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. 

THIS SOFTWARE, AND ANY ACCOMPANYING DOCUMENTATION, IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE OR ANY OF ITS ACCOMPANYING DOCUMENTATION, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

