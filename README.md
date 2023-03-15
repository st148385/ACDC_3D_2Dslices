Train 2D models for 3D segmentation using 2D Slices of 3D ACDC NIfTI files (.nii.gz)

First 50 ACDC 3D files are used for Training (35), Validation (5), Testing (10).



  

Alternatively use the [nnU-Net by Fabian Isensee et al.](https://github.com/MIC-DKFZ/nnUNet) as a baseline result (file: [2D_nnU_Net_ACDC_TopBaseline.ipynb](https://github.com/st148385/ACDC_3D_2Dslices/blob/main/2D_nnU_Net_ACDC_TopBaseline.ipynb)). There the first 40 files are used for a training with 5-fold cross validation (split of 8 validation and 32 training cases)

  

ACDC Classes in images by color:  
Yellow: LV cavity       &nbsp;&nbsp;&nbsp;// Value = 3; inside  
Green: Myocardium       &nbsp;&nbsp;&nbsp;// Value = 2; transition  
Blue: RV cavity         &nbsp;&nbsp;&nbsp;// Value = 1; outside  
(Purple: Background)  
