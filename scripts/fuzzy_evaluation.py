import glob
import os

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm


def getClass(test_df,class_list,source="target",threshold=None):
    if source == "target":
        suffix = "_true"
    elif source == "prediction":
        suffix = "_pred"
    else:
        raise ValueError("source must be 'target' or 'prediction'")

    crop_types_single = [col.replace(suffix,"") for col in test_df.columns if col.endswith(suffix)][:-2]
    if threshold is None:
        threshold = 1 / (len(crop_types_single) -1)

    #in the target labels, any value above 0 is considered present
    if source == "target":
        threshold = 0
    target_classes = []
    for i, row in test_df.iterrows():
        #for which columns is the value higher than threshold
        classes = []
        for crop in crop_types_single:
            if row[f'{crop}{suffix}'] > threshold:
                classes.append(crop)
        #create a string with the classes separated by x
        target_class = "x".join(classes)

        if "x" in target_class:
            #check if the class is in the acceptable list
            if target_class not in class_list:
                target_class = "other"

        target_classes.append(target_class)
    return target_classes

def getOA_threshold(predictions, class_list, threshold=0.5):
    targets = predictions.loc[predictions["source"]== "target",].reset_index(drop=True)
    predictions = predictions.loc[predictions["type"]== "prediction",].reset_index(drop=True)

    test = predictions.merge(targets, on="sample_id", suffixes=('_pred', '_true'))

    test["target_class"] = getClass(test,class_list,source="target",threshold=threshold)
    test["predicted_class"] = getClass(test,class_list,source="prediction",threshold=threshold)

    report = classification_report(test["target_class"], test["predicted_class"], output_dict=True)
    OA = report['accuracy']
    return OA

def getF1_threshold(predictions,class_list,threshold=0.5):
    targets = predictions.loc[predictions["source"]== "target",].reset_index(drop=True)
    predictions = predictions.loc[predictions["type"]== "prediction",].reset_index(drop=True)

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
    return average_F1



def determineOptimalThreshold(predictions, class_list, indicator = "average_F1",makePlot=False,output_folder=None):

    targets = predictions.loc[predictions["source"]== "target",].reset_index(drop=True)
    predictions = predictions.loc[predictions["type"]== "prediction",].reset_index(drop=True)

    test = predictions.merge(targets, on="sample_id", suffixes=('_pred', '_true'))

    thresholds = [round(x * 0.01,2) for x in range(0,101)]

    if indicator == "OA":
        indicator_values = []
    elif indicator == "average_F1":
        indicator_values: dict[str, list[float]] = {cls: [] for cls in class_list}
    else:
        raise ValueError("indicator must be 'OA' or 'average_F1'")

    for threshold in tqdm(thresholds,desc="Determining optimal threshold"):
        test["target_class"] = getClass(test,class_list,source="target",threshold=threshold)
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

def createConfusionMatrix_threshold(predictions, output_folder, class_list, threshold=0.5):

    targets = predictions.loc[predictions["source"]== "target",].reset_index(drop=True)
    predictions = predictions.loc[predictions["type"]== "prediction",].reset_index(drop=True)

    test = predictions.merge(targets, on="sample_id", suffixes=('_pred', '_true'))

    test["target_class"] = getClass(test,class_list,source="target",threshold=threshold)
    test["predicted_class"] = getClass(test,class_list,source="prediction",threshold=threshold)

    cm = confusion_matrix(test["target_class"], test["predicted_class"], labels=test["target_class"].unique())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test["target_class"].unique())
    disp.plot(xticks_rotation='vertical')
    #add title with threshold value
    plt.title(f'Confusion Matrix (threshold={threshold})')
    plt.show()
    # Save confusion matrix
    disp.figure_.savefig(os.path.join(output_folder, f'confusion_matrix_thr_{threshold}.png'), bbox_inches='tight')


if __name__ == "__main__":


    folder = '/vitodata/worldcereal/data/COP4GEOGLAM/mozambique/fuzzy_test/'

    pred_file = glob.glob(os.path.join(folder, 'predictions*.parquet'))[0]

    predictions = pd.read_parquet(pred_file)

    class_list = [
            "maize",
            "soybean",
            "sesame",
            "sweet_potato",
            "cassava",
            "pigeon pea",
            "rice",
            "other",
            "maizexcassava",
            "cassavaxpigeon pea",
            "maizexcassavaxpigeon pea",
        ]

    #OA analysis
    #OA_threshold, OA = determineOptimalThreshold(predictions,class_list,"OA",makePlot=True,output_folder=folder)

    #average F1 analysis
    F1_threshold, F1 = determineOptimalThreshold(predictions,class_list,"average_F1",makePlot=True,output_folder=folder)


    #Create confusion matrix for the best F1 threshold
    createConfusionMatrix_threshold(predictions,folder, class_list, threshold=F1_threshold)
