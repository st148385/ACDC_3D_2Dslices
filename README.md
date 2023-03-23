Train 2D models for 3D segmentation using 2D Slices of 3D ACDC NIfTI files (.nii.gz)

The ACDC files have to be downloaded to a Google Drive account.

Using [this file](https://github.com/st148385/ACDC_3D_2Dslices/blob/main/lr_Scheduling_niftiSave_Unet_ACDC_3D_2DsliceTraining_segmentation.ipynb) for training and evaluation, the first 50 ACDC 3D files are used for a split of Training (35), Validation (5) and Testing (10).



  

Alternatively use the [nnU-Net by Fabian Isensee et al.](https://github.com/MIC-DKFZ/nnUNet) as a baseline result (file: [2D_nnU_Net_ACDC_TopBaseline.ipynb](https://github.com/st148385/ACDC_3D_2Dslices/blob/main/2D_nnU_Net_ACDC_TopBaseline.ipynb)). There the first 40 files are used for a training with 5-fold cross validation (split of 8 validation and 32 training cases). To obtain a .csv of the evaluation metrics and produce a boxplot with some metrics in the .csv file afterwards, [this file](https://github.com/st148385/ACDC_3D_2Dslices/blob/main/ACDC_OfficialEvaluationMetrics.py) can be used.



ACDC Classes in images by color:  
Yellow: LV cavity       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; // Value = 3; inside  
Green: Myocardium       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; // Value = 2; transition  
Blue: RV cavity         &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; // Value = 1; outside  
(Purple: Background; Value = 0)  
