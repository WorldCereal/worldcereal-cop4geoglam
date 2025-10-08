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


def getClass(test_df,source="target",threshold=None,acceptable_list = ["maizexcassava","cassavaxpigeon pea","maizexcassavaxpigeon pea"]):
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
            if target_class not in acceptable_list:
                target_class = "other"

        target_classes.append(target_class)
    return target_classes


folder = '/vitodata/worldcereal/data/COP4GEOGLAM/mozambique/fuzzy_test/'

pred_file = glob.glob(os.path.join(folder, 'predictions*.parquet'))[0]

predictions = pd.read_parquet(pred_file)

targets = predictions.loc[predictions["source"]== "target",].reset_index(drop=True)
predictions = predictions.loc[predictions["type"]== "prediction",].reset_index(drop=True)

test = predictions.merge(targets, on="sample_id", suffixes=('_pred', '_true'))

test["target_class"] = getClass(test,source="target")
test["predicted_class"] = getClass(test,source="prediction")

#create an analysis for different thresholds between 0-1 in steps of 0.01

thresholds = [round(x * 0.01,2) for x in range(0,101)]

OA_values = []

#do a F1 for the following classes:
F1_classes = [
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

F1_values: dict[str, list[float]] = {cls: [] for cls in F1_classes}

for threshold in tqdm(thresholds,desc="Determining optimal threshold"):
    test["target_class"] = getClass(test,source="target",threshold=threshold)
    test["predicted_class"] = getClass(test,source="prediction",threshold=threshold)
    report = classification_report(test["target_class"], test["predicted_class"], output_dict=True)
    OA = report['accuracy']
    OA_values.append(OA)
    for cls in F1_classes:
        if cls in report:
            F1 = report[cls]['f1-score']
        else:
            F1 = 0.0
        F1_values[cls].append(F1)

# Save OA values to a dataframe
OA_df = pd.DataFrame({'threshold': thresholds, 'OA': OA_values})
#make a line plot of OA vs threshold
plt.figure()
plt.plot(OA_df['threshold'], OA_df['OA'])
plt.xlabel('Threshold')
plt.ylabel('Overall Accuracy (OA)')
plt.title('Overall Accuracy vs Threshold')
plt.grid()
plt.savefig(os.path.join(folder, 'OA_vs_threshold.png'), bbox_inches='tight')
plt.show()

#average F1 across all classes
avg_F1_values = []
for i in range(len(thresholds)):
    avg_F1 = sum(F1_values[cls][i] for cls in F1_classes) / len(F1_classes)
    avg_F1_values.append(avg_F1)

#identify the threshold with the highest average F1
best_threshold_index = avg_F1_values.index(max(avg_F1_values))

#create plot for F1 values
F1_df = pd.DataFrame({'threshold': thresholds})
for cls in F1_classes:
    F1_df[cls] = F1_values[cls]
#make a line plot of F1 vs threshold for each class
plt.figure()
for cls in F1_classes:
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
plt.savefig(os.path.join(folder, 'F1_vs_threshold.png'), bbox_inches='tight')
plt.show()

#create confusion matrix for the best threshold
best_threshold = thresholds[best_threshold_index]
test["target_class"] = getClass(test,source="target",threshold=best_threshold)
test["predicted_class"] = getClass(test,source="prediction",threshold=best_threshold)
cm = confusion_matrix(test["target_class"], test["predicted_class"], labels=test["target_class"].unique())
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test["target_class"].unique())
disp.plot(xticks_rotation='vertical')
plt.show()
# Save confusion matrix
disp.figure_.savefig(os.path.join(folder, 'confusion_matrix_best_threshold.png'), bbox_inches='tight')

report = classification_report(test["target_class"], test["predicted_class"])
print(report)
#print confusion matrix that also shows the predicted classes that are not in the true classes
