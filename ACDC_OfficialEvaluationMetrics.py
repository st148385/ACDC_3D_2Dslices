# Editiert zur zusätzlichen Erstellung eines Boxplots aus dem .csv file
"""
author: Clément Zotti (clement.zotti@usherbrooke.ca)
date: April 2017

DESCRIPTION :
The script provide helpers functions to handle nifti image format:
    - load_nii()
    - save_nii()

to generate metrics for two images:
    - metrics()

And it is callable from the command line (see below).
Each function provided in this script has comments to understand
how they works.

HOW-TO:

This script was tested for python 3.4.

First, you need to install the required packages with
    pip install -r requirements.txt

After the installation, you have two ways of running this script:
    1) python metrics.py ground_truth/patient001_ED.nii.gz prediction/patient001_ED.nii.gz
    2) python metrics.py ground_truth/ prediction/

The first option will print in the console the dice and volume of each class for the given image.
The second option wiil ouput a csv file where each images will have the dice and volume of each class.


Link: http://acdc.creatis.insa-lyon.fr

"""

# Beispiel cmd Aufruf: "python filename.py <gtLabel> <predictedLabel>"
# python ACDC_OfficialEvaluationMetrics.py output/patient041_gtLabel_2D/patient041_frame01_gt_2D.nii.gz output/patient041_frame01_2D/patient041_frame01_gt_2D.nii.gz

import os
from glob import glob
import time
import re
import argparse
import nibabel as nib
import pandas as pd
from medpy.metric.binary import hd, dc
import numpy as np
import csv  # Newly added -> To read the output .csv file
import matplotlib.pyplot as plt    # Newly added -> To plot boxplots


HEADER = ["Name", "Dice LV", "Volume LV", "Err LV(ml)",
          "Dice RV", "Volume RV", "Err RV(ml)",
          "Dice MYO", "Volume MYO", "Err MYO(ml)"]

#
# Utils functions used to sort strings into a natural order
#
def conv_int(i):
    return int(i) if i.isdigit() else i


def natural_order(sord):
    """
    Sort a (list,tuple) of strings into natural order.

    Ex:

    ['1','10','2'] -> ['1','2','10']

    ['abc1def','ab10d','b2c','ab1d'] -> ['ab1d','ab10d', 'abc1def', 'b2c']

    """
    if isinstance(sord, tuple):
        sord = sord[0]
    return [conv_int(c) for c in re.split(r'(\d+)', sord)]


#
# Utils function to load and save nifti files with the nibabel package
#
def load_nii(img_path):
    """
    Function to load a 'nii' or 'nii.gz' file, The function returns
    everyting needed to save another 'nii' or 'nii.gz'
    in the same dimensional space, i.e. the affine matrix and the header

    Parameters
    ----------

    img_path: string
    String with the path of the 'nii' or 'nii.gz' image file name.

    Returns
    -------
    Three element, the first is a numpy array of the image values,
    the second is the affine transformation of the image, and the
    last one is the header of the image.
    """
    nimg = nib.load(img_path)
    return nimg.get_fdata(), nimg.affine, nimg.header


def save_nii(img_path, data, affine, header):
    """
    Function to save a 'nii' or 'nii.gz' file.

    Parameters
    ----------

    img_path: string
    Path to save the image should be ending with '.nii' or '.nii.gz'.

    data: np.array
    Numpy array of the image data.

    affine: list of list or np.array
    The affine transformation to save with the image.

    header: nib.Nifti1Header
    The header that define everything about the data
    (pleasecheck nibabel documentation).
    """
    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)


#
# Functions to process files, directories and metrics
#
def metrics(img_gt, img_pred, voxel_size):
    """
    Function to compute the metrics between two segmentation maps given as input.

    Parameters
    ----------
    img_gt: np.array
    Array of the ground truth segmentation map.

    img_pred: np.array
    Array of the predicted segmentation map.

    voxel_size: list, tuple or np.array
    The size of a voxel of the images used to compute the volumes.

    Return
    ------
    A list of metrics in this order, [Dice LV, Volume LV, Err LV(ml),
    Dice RV, Volume RV, Err RV(ml), Dice MYO, Volume MYO, Err MYO(ml)]
    """

    if img_gt.ndim != img_pred.ndim:
        raise ValueError("The arrays 'img_gt' and 'img_pred' should have the "
                         "same dimension, {} against {}".format(img_gt.ndim,
                                                                img_pred.ndim))

    res = []
    # Loop on each classes of the input images
    for c in [3, 1, 2]:
        # Copy the gt image to not alterate the input
        gt_c_i = np.copy(img_gt)
        gt_c_i[gt_c_i != c] = 0

        # Copy the pred image to not alterate the input
        pred_c_i = np.copy(img_pred)
        pred_c_i[pred_c_i != c] = 0

        # Clip the value to compute the volumes
        gt_c_i = np.clip(gt_c_i, 0, 1)
        pred_c_i = np.clip(pred_c_i, 0, 1)

        # Compute the Dice
        dice = dc(gt_c_i, pred_c_i)

        # Compute volume
        volpred = pred_c_i.sum() * np.prod(voxel_size) / 1000.
        volgt = gt_c_i.sum() * np.prod(voxel_size) / 1000.

        res += [dice, volpred, volpred-volgt]

    return res


def compute_metrics_on_files(path_gt, path_pred):
    """
    Function to give the metrics for two files

    Parameters
    ----------

    path_gt: string
    Path of the ground truth image.

    path_pred: string
    Path of the predicted image.
    """
    gt, _, header = load_nii(path_gt)
    pred, _, _ = load_nii(path_pred)
    zooms = header.get_zooms()

    name = os.path.basename(path_gt)
    name = name.split('.')[0]
    res = metrics(gt, pred, zooms)
    res = ["{:.3f}".format(r) for r in res]

    formatting = "{:>14}, {:>7}, {:>9}, {:>10}, {:>7}, {:>9}, {:>10}, {:>8}, {:>10}, {:>11}"
    print(formatting.format(*HEADER))
    print(formatting.format(name, *res))


def compute_metrics_on_directories(dir_gt, dir_pred):
    """
    Function to generate a csv file for each images of two directories.

    Parameters
    ----------

    path_gt: string
    Directory of the ground truth segmentation maps.

    path_pred: string
    Directory of the predicted segmentation maps.
    """
    lst_gt = sorted(glob(os.path.join(dir_gt, '*')), key=natural_order)
    lst_pred = sorted(glob(os.path.join(dir_pred, '*')), key=natural_order)

    res = []
    for p_gt, p_pred in zip(lst_gt, lst_pred):
        if os.path.basename(p_gt) != os.path.basename(p_pred):
            raise ValueError("The two files don't have the same name"
                             " {}, {}.".format(os.path.basename(p_gt),
                                               os.path.basename(p_pred)))

        gt, _, header = load_nii(p_gt)
        pred, _, _ = load_nii(p_pred)
        zooms = header.get_zooms()
        res.append(metrics(gt, pred, zooms))

    lst_name_gt = [os.path.basename(gt).split(".")[0] for gt in lst_gt]
    res = [[n,] + r for r, n in zip(res, lst_name_gt)]
    df = pd.DataFrame(res, columns=HEADER)
    df.to_csv("results_{}.csv".format(time.strftime("%Y%m%d_%H%M%S")), index=False)

def main(path_gt, path_pred):
    """
    Main function to select which method to apply on the input parameters.
    """
    if os.path.isfile(path_gt) and os.path.isfile(path_pred):
        compute_metrics_on_files(path_gt, path_pred)
    elif os.path.isdir(path_gt) and os.path.isdir(path_pred):
        compute_metrics_on_directories(path_gt, path_pred)
    else:
        raise ValueError(
            "The paths given needs to be two directories or two files.")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="Script to compute ACDC challenge metrics.")
    # parser.add_argument("GT_IMG", type=str, help="Ground Truth image")
    # parser.add_argument("PRED_IMG", type=str, help="Predicted image")
    # args = parser.parse_args()
    # main(args.GT_IMG, args.PRED_IMG)
    GT_IMG = 'output/ACDC_GroundTruth'
    PRED_IMG = 'output/nnUNet_Output'
    # main(GT_IMG, PRED_IMG)    # Get the .csv file by comparing all nifti files in the folder GT_IMG to all nifti files in the folder PRED_IMG

    # Beispiel cmd Aufruf: "python filename.py <gtLabel> <predictedLabel>"
    # python ACDC_OfficialEvaluationMetrics.py output/patient041_gtLabel_2D/patient041_frame01_2D_seg.nii.gz output/patient041_frame01_2D/patient041_frame01_gt_2D.nii.gz

    ## Get averages from .csv file and plot boxplots
    results = []

    with open("results_20230315_145308.csv") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            results.append(row)

    ## Reihenfolge im Excel file: LV, RV, Myo       // d. h. Segmentation mask value Reihenfolge = [3, 1, 2]
    # results[x][1] = Dice LV cavity
    # results[x][4] = Dice RV cavity
    # results[x][7] = Dice Myocardium

    dice_metric_matrix = np.zeros([len(results) - 1, 3])

    for i in range(len(results)-1):
        dice_metric_matrix[i][2] = results[i+1][1]      # results[x][1] = LV dice -> das hat in der Maske den Wert 3 -> letzter boxplot [i][2]
        dice_metric_matrix[i][0] = results[i+1][4]      # results[x][4] = RV dice -> Wert in Maske = 1 -> erster boxplot [i][0]
        dice_metric_matrix[i][1] = results[i+1][7]      # results[x][7] = Myo dice -> Wert in Maske = 2 -> mittlerer boxplot [i][1]


    RV_elements_baseline = dice_metric_matrix[:,0]
    Myo_elements_baseline = dice_metric_matrix[:,1]
    LV_elements_baseline = dice_metric_matrix[:,2]

    print("Mean dice for class 'RV cavity' =", np.mean(RV_elements_baseline))
    print("Mean dice for class 'Myocardium' =", np.mean(Myo_elements_baseline))
    print("Mean dice for class 'LV cavity' =", np.mean(LV_elements_baseline))

    print("Overall mean dice for all three classes =", np.mean(dice_metric_matrix))

    ### plot font size
    import matplotlib
    font = {'family': 'serif',
            'size': 15}
    matplotlib.rc('font', **font)
    ###

    # Boxplot of .csv file
    fig, ax = plt.subplots()
    ax.boxplot([dice_metric_matrix[:,0], dice_metric_matrix[:,1], dice_metric_matrix[:,2]])  #
    ax.set_xticklabels(['RV cavity', 'Myocardium',
                        'LV cavity'])  # "0": "voxels in the background", "1": "voxels in the RV cavity", "2": "voxels in the myocardium", "3": "voxels in the LV cavity"
    ax.set_ylabel('Dice score of ten 3D masks during test')
    ax.set_title("Baseline")
    plt.tight_layout()
    #ax.set_ylim(0.74,0.98)
    PlotSaveLocation = "output/Boxplot_10Test3Dimages_nnUNet_baseline.pdf"
    # plt.savefig(PlotSaveLocation + ".pdf")
    plt.savefig(PlotSaveLocation)
    print(f"Saved to {PlotSaveLocation}")
    plt.show()


    # Individual boxplot (Vorsicht! So ist Segresnet_data eine list of 10 lists with length 3 each. dice_metric_matrix ist aber ein numpyarray mit Shape (10,3).)
    Segresnet_data =   [[0.8103033,  0.7533007,  0.9147179],
                        [0.8973934,  0.78224194, 0.9498299],
                        [0.871188, 0.7429745, 0.92461854],
                        [0.79385495, 0.7295972,  0.8955965],
                        [0.8657085,   0.72332686,  0.91936976],
                        [0.839245,   0.77262354, 0.9099823],
                        [0.46524727,     0.69121677,    0.89069444],
                        [0.73333067, 0.7993905,  0.9100484],
                        [0.7400252,     0.78283584,     0.922335],
                        [0.7452444,  0.7295882,  0.9059352]]

    RV_elements = [lists[0] for lists in Segresnet_data]
    Myo_elements = [lists[1] for lists in Segresnet_data]
    LV_elements = [lists[2] for lists in Segresnet_data]


    fig, ax = plt.subplots()
    ax.boxplot([RV_elements, Myo_elements, LV_elements])
    ax.set_xticklabels(['RV cavity', 'Myocardium',
                        'LV cavity'])  # "0": "voxels in the background", "1": "voxels in the RV cavity", "2": "voxels in the myocardium", "3": "voxels in the LV cavity"
    ax.set_ylabel('Dice score of ten 3D masks during test')
    ax.set_title('New Implementation (Segresnet)')
    plt.tight_layout()
    #ax.set_ylim(0.735, 0.99)
    PlotSaveLocation2 = "output/Boxplot_10Test3Dimages_ownImplementation_Segresnet.pdf"
    # plt.savefig(PlotSaveLocation2 + ".pdf")
    plt.savefig(PlotSaveLocation2)
    print(f"Saved to {PlotSaveLocation2}")
    plt.show()



    print("Mean dice for class 'RV cavity' =", np.mean(RV_elements))
    print("Mean dice for class 'Myocardium' =", np.mean(Myo_elements))
    print("Mean dice for class 'LV cavity' =", np.mean(LV_elements))

    print("Overall mean dice for all three classes =", np.mean(Segresnet_data))