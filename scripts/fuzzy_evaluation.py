import glob
import os
import warnings

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from matplotlib import pyplot as plt
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def getClass(test_df, class_list, source="target", threshold=None, assignOther=True,otherValue = "other"):
    """
    Determines the class for each row in the DataFrame based on thresholds.

    Args:
        test_df (pd.DataFrame): DataFrame containing the data.
        class_list (list): List of valid class names.
        source (str): Either "target" or "prediction" to specify the source of data.
        thresholds (float or list): A single threshold value or a list of thresholds for each class.

    Returns:
        list: List of determined classes for each row.
    """
    if source == "target":
        suffix = "_true"
    elif source == "prediction":
        suffix = "_pred"
    else:
        raise ValueError("source must be 'target' or 'prediction'")

    crop_types_single = [col.replace(suffix, "") for col in test_df.columns if col.endswith(suffix)]
    #only select the crop types that are in the class_list
    crop_types_single = [crop for crop in crop_types_single if crop in class_list]

    # If thresholds is None, set a default threshold for all classes
    if threshold is None:
        threshold = [1 / (len(crop_types_single) - 1)] * len(crop_types_single)
    elif isinstance(threshold, (int, float)):
        threshold = [threshold] * len(crop_types_single)
    elif len(threshold) != len(crop_types_single):
        raise ValueError("Thresholds must have the same length as the number of classes.")

    # In the target labels, any value above 0 is considered present
    if source == "target":
        threshold = [0] * len(crop_types_single)

    target_classes = []
    for _, row in test_df.iterrows():
        # Determine which columns have values higher than their respective thresholds
        classes = []
        for crop, thr in zip(crop_types_single, threshold):
            if row[f'{crop}{suffix}'] > thr:
                classes.append(crop)

        # Create a string with the classes separated by 'x'
        target_class = "x".join(classes)

        if assignOther:
            if "x" in target_class:
                # Check if the class is in the acceptable list
                if target_class not in class_list:
                    target_class = otherValue

        target_classes.append(target_class)

    return target_classes

def getOA_threshold(predictions, class_list, threshold=0):
    targets = predictions.loc[predictions["source"]== "target",].reset_index(drop=True)
    predictions = predictions.loc[predictions["source"]== "prediction",].reset_index(drop=True)

    test = predictions.merge(targets, on="sample_id", suffixes=('_pred', '_true'))

    test["target_class"] = getClass(test,class_list,source="target",threshold=threshold)
    test["predicted_class"] = getClass(test,class_list,source="prediction",threshold=threshold)

    report = classification_report(test["target_class"], test["predicted_class"], output_dict=True)
    OA = report['accuracy']
    return OA

def getF1_threshold(predictions,class_list,threshold=0.5):
    targets = predictions.loc[predictions["source"]== "target",].reset_index(drop=True)
    predictions = predictions.loc[predictions["source"]== "prediction",].reset_index(drop=True)

    test = predictions.merge(targets, on="sample_id", suffixes=('_pred', '_true'))

    test["target_class"] = getClass(test,class_list,source="target",threshold=threshold)
    test["predicted_class"] = getClass(test,class_list,source="prediction",threshold=threshold)

    report = classification_report(test["target_class"], test["predicted_class"], output_dict=True)
    F1_values = {}
    if class_list is None:
        class_list = test["target_class"].unique().tolist()
    for cls in class_list:
        if cls in report:
            F1 = report[cls]['f1-score']
        else:
            F1 = 0.0
        F1_values[cls] = F1
    average_F1 = sum(F1_values[cls] for cls in class_list) / len(class_list)
    print(f"{threshold}: {average_F1}")
    return average_F1

def determineOptimalThreshold(predictions, class_list, indicator = "average_F1", makePlot=False, output_folder=None):

    if 'type' in predictions.columns:
        targets = predictions.loc[predictions["source"]== "target",].reset_index(drop=True)
        predictions = predictions.loc[predictions["type"]== "prediction",].reset_index(drop=True)
    else:
        targets = predictions.loc[predictions["source"]== "target",].reset_index(drop=True)
        predictions = predictions.loc[predictions["source"]== "prediction",].reset_index(drop=True)

    test = predictions.merge(targets, on="sample_id", suffixes=('_pred', '_true'))

    thresholds = [round(x * 0.01,2) for x in range(0,101)]

    if indicator == "OA":
        indicator_values = []
    elif indicator == "average_F1":
        indicator_values: dict[str, list[float]] = {cls: [] for cls in class_list}
    else:
        raise ValueError("indicator must be 'OA' or 'average_F1'")

    for threshold in tqdm(thresholds,desc="Determining optimal threshold"):
        test["target_class"] = getClass(test,class_list,source="target",threshold=0)
        test["predicted_class"] = getClass(test,class_list,source="prediction",threshold=threshold)
        report = classification_report(test["target_class"], test["predicted_class"], output_dict=True)
        if indicator == "OA":
            OA = report['accuracy']
            indicator_values.append(OA)

            OA_values = indicator_values


        elif indicator == "average_F1":
            for cls in class_list:
                if cls in report:
                    F1 = report[cls]['f1-score']
                else:
                    F1 = 0.0
                indicator_values[cls].append(F1)

            #identify the threshold with the highest average F1

        else:
            raise ValueError("indicator must be 'OA' or 'average_F1'")

    if indicator == "OA":
        best_OA_index = OA_values.index(max(OA_values))
        if makePlot and output_folder is not None:

            # Save OA values to a dataframe
            OA_df = pd.DataFrame({'threshold': thresholds, 'OA': OA_values})
            #make a line plot of OA vs threshold
            plt.figure()
            plt.plot(OA_df['threshold'], OA_df['OA'])
            plt.axvline(x=thresholds[best_OA_index], color='red', linestyle='--', label='best threshold')
                    #add text to indicate the best threshold
            plt.text(thresholds[best_OA_index], 0.5, f'Best threshold: {thresholds[best_OA_index]}', color='red')
            plt.xlabel('Threshold')
            plt.ylabel('Overall Accuracy (OA)')
            plt.title('Overall Accuracy vs Threshold')
            plt.grid()
            plt.savefig(os.path.join(output_folder, 'OA_vs_threshold.png'), bbox_inches='tight')
            plt.show()

        return thresholds[best_OA_index], OA_values[best_OA_index]

    elif indicator == "average_F1":

        avg_F1_values = []
        for i in range(len(thresholds)):
            avg_F1 = sum(indicator_values[cls][i] for cls in class_list) / len(class_list)
            avg_F1_values.append(avg_F1)

        best_threshold_index = avg_F1_values.index(max(avg_F1_values))

        if makePlot and output_folder is not None:
            #create plot for F1 values
            F1_df = pd.DataFrame({'threshold': thresholds})
            for cls in class_list:
                F1_df[cls] = indicator_values[cls]
            #make a line plot of F1 vs threshold for each class
            plt.figure()
            for cls in class_list:
                plt.plot(F1_df['threshold'], F1_df[cls], label=cls)
            #also plot average F1
            plt.plot(F1_df['threshold'], avg_F1_values, label='average', linewidth=2, color='black')
            #highlight the best threshold
            plt.axvline(x=thresholds[best_threshold_index], color='red', linestyle='--', label='best threshold')
            #add text to indicate the best threshold
            plt.text(thresholds[best_threshold_index], 0.5, f'Best threshold: {thresholds[best_threshold_index]}', color='red')
            plt.xlabel('Threshold')
            plt.ylabel('F1 Score')
            plt.title('F1 Score vs Threshold')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid()
            plt.savefig(os.path.join(output_folder, 'F1_vs_threshold.png'), bbox_inches='tight')
            plt.show()
        return thresholds[best_threshold_index], avg_F1_values[best_threshold_index]

    else:
        raise ValueError("indicator must be 'OA' or 'average_F1'")

def createConfusionMatrix_threshold(predictions, output_folder, class_list, threshold=0.5,run_name=None):

    if 'type' in predictions.columns:
        targets = predictions.loc[predictions["source"]== "target",].reset_index(drop=True)
        predictions = predictions.loc[predictions["type"]== "prediction",].reset_index(drop=True)
    else:
        targets = predictions.loc[predictions["source"]== "target",].reset_index(drop=True)
        predictions = predictions.loc[predictions["source"]== "prediction",].reset_index(drop=True)

    test = predictions.merge(targets, on="sample_id", suffixes=('_pred', '_true'))

    test["target_class"] = getClass(test,class_list,source="target",threshold=0)
    test["predicted_class"] = getClass(test,class_list,source="prediction",threshold=threshold)

    F1 = getF1_threshold(predictions,class_list,threshold=threshold)

    cm = confusion_matrix(test["target_class"], test["predicted_class"], labels=test["predicted_class"].unique())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test["target_class"].unique())
    disp.plot(xticks_rotation='vertical')
    #add title with threshold value
    plt.title(f'Confusion Matrix: F1: {F1:.2f}')
    plt.show()
    # Save confusion matrix
    disp.figure_.savefig(os.path.join(output_folder, f'confusion_matrix_thr_{threshold}_run_{run_name}.png'), bbox_inches='tight')

def applyThreshold(nc_file,class_list,output_folder,thresholds,otherValue = "other"):

    os.makedirs(output_folder, exist_ok=True)

    if nc_file.split('.')[-1] == 'tif':
        applyThreshold_tif(nc_file,class_list,output_folder,thresholds,otherValue=otherValue)
    else:
        #Load the NC file
        ds = xr.open_dataset(nc_file)

        single_class_list = [cls for cls in class_list if 'x' not in cls]
        single_class_list = [cls for cls in single_class_list if cls in ds.data_vars]

        if isinstance(thresholds, (int, float)):
            thresholds = [thresholds] * len(single_class_list)

        #Create an empty array to store the temp (type = string)
        temp = np.full(ds[single_class_list[0]].shape, '', dtype=object)
        for i, crop in enumerate(single_class_list):
            #make sure that 65535 values are set to 0
            ds[crop] = ds[crop].where(ds[crop] != 65535, 0)
            crop_mask = ds[crop] > thresholds[i]
            temp = np.where(crop_mask, np.where(temp == '', crop, temp + 'x' + crop), temp)

        class_list.append("")
        #check whether the class is in the acceptable list, if not set to "other"
        temp = np.where(np.isin(temp, class_list), temp, "other")

        # give each unique value an integer according to its index in class_list
        int_map = {cls: idx for idx, cls in enumerate(class_list)}
        int_array = np.vectorize(int_map.get)(temp)
        #set all values that are "" to np.nan
        int_array = np.where(temp == "", np.nan, int_array)

        #define the output file name
        output_file = os.path.join(output_folder, os.path.basename(nc_file).replace('.nc', f'_thr{thresholds[0]}_classified.nc'))
        #save as tif if the input is a tif
        if os.path.endswith(nc_file, '.tif'):
            output_file = output_file.replace('.nc', '.tif')

        # copy the ds to a new dataset to avoid modifying the original dataset
        ds_out = ds.copy()
        # Add the integer array as a new variable to the dataset
        ds_out['classification'] = ds_out[class_list[0]].copy(data=int_array)
        #save the ds_out
        ds_out['classification'].attrs['long_name'] = 'Crop type classification'
        ds_out['classification'].attrs['classes'] = ','.join(class_list)
        ds_out['classification'].attrs['thresholds'] = ','.join([str(thr) for thr in thresholds])
        ds_out['classification'].attrs['nodata'] = np.nan
        #save as nc file
        ds_out.to_netcdf(output_file)
        ds_out.close()

def applyThreshold_tif(tif_file,class_list,output_folder,thresholds,otherValue = "other"):

    output_file = os.path.join(output_folder, os.path.basename(tif_file).replace('.tif', f'_thr{thresholds}_classified.tif'))

    #Load the tif file
    with rasterio.open(tif_file) as src:
        data = src.read()
        profile = src.profile

    single_class_list = [cls for cls in class_list if 'x' not in cls]

    if isinstance(thresholds, (int, float)):
        thresholds = [thresholds] * len(single_class_list)

    #Create an empty array to store the temp (type = string)
    temp = np.full(data[0].shape, '', dtype=object)
    no_crop_mask = data[0] > 2
    for i, crop in enumerate(single_class_list):
        #make sure that 65535 values are set to 0
        data[i] = np.where(data[i] > 2, 0, data[i])
        crop_mask = data[i] > thresholds[i]
        temp = np.where(crop_mask, np.where(temp == '', crop, temp + 'x' + crop), temp)

    class_list_cop = class_list.copy()
    if otherValue not in class_list_cop:
        class_list_cop.append(otherValue)
    class_list_cop.append("")
    #check whether the class is in the acceptable list, if not set to "other"
    temp = np.where(np.isin(temp, class_list_cop), temp, otherValue)

    # give each unique value an integer according to its index in class_list
    int_map = {cls: idx for idx, cls in enumerate(class_list_cop)}
    int_array = np.vectorize(int_map.get)(temp)
    # Set all values that are "" to -9999 (nodata value for int16)
    int_array = np.where(temp == "", -9999, int_array)
    int_array = int_array.astype(np.int16)

    # Reshape to 1, height, width
    int_array = int_array.reshape(1, int_array.shape[0], int_array.shape[1])
    int_array = np.where(no_crop_mask, -9999, int_array)

    # Define the output file name
    output_file = os.path.join(output_folder, os.path.basename(tif_file).replace('.tif', f'_thr{thresholds}_classified.tif'))

    # Update the profile with the correct nodata value
    profile.update(
        dtype=np.int16,
        count=1,
        compress='lzw',
        nodata=-9999  # Set nodata value to -9999
    )

    #link the values to the class names in the metadata

    # Write the classified data to a new GeoTIFF file
    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(int_array)
        dst.update_tags(
            1,
            long_name='Crop type classification',
            classes=','.join(class_list),
            thresholds=','.join([str(thr) for thr in thresholds]),
            nodata=-9999  # Ensure nodata value matches
        )


if __name__ == "__main__":

    main_folder = "/vitodata/worldcereal/data/COP4GEOGLAM/mozambique/"

    production = 'v1_croptype'

    nc_folder = os.path.join(main_folder,"production",production,"raw")
    folder = os.path.join(main_folder,"fuzzy_test")
    os.makedirs(folder, exist_ok=True)

    pred_file = glob.glob(os.path.join(folder, 'predictions_presto_run=202510151422.parquet'))[0]

    predictions = pd.read_parquet(pred_file)

    class_list = [
            "maize",
            "rice",
            "soybean",
            "sesame",
            "cassava",
            "sweet_potato",
            "pigeon_pea",
            "other",
            "maizexcassava",
            "cassavaxpigeon_pea",
            "maizexcassavaxpigeon_pea"
        ]

    class_list_single = [
            "maize",
            "rice",
            "soybean",
            "sesame",
            "cassava",
            "sweet_potato",
            "pigeon_pea",
            "other",
        ]

    #OA analysis
    #OA_threshold, OA = determineOptimalThreshold(predictions,class_list,"OA",makePlot=True,output_folder=folder)

    #average F1 analysis
    #F1_threshold, F1 = determineOptimalThreshold(predictions,class_list,"average_F1",makePlot=True,output_folder=folder)

    F1_threshold = [0.18,0.24,0.16,0.25,0.21,0.15,0.24,0.31]

    #Create confusion matrix for the best F1 threshold
    createConfusionMatrix_threshold(predictions,folder, class_list, threshold=F1_threshold, run_name = "v3")

    tif_files = glob.glob(os.path.join(nc_folder,"*",'croptype*.tif'))

    #for tif_file in tqdm(tif_files,desc="Processing nc files"):
        #applyThreshold(tif_file,class_list,output_folder=os.path.join(folder,production),thresholds=F1_threshold,otherValue = "other_mix")
