import json
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd
import torch
from loguru import logger
from prometheo.predictors import NODATAVALUE, Predictors
from prometheo.utils import device
from worldcereal.utils.refdata import map_classes
from worldcereal.utils.timeseries import process_parquet

from worldcereal_cop4geoglam.datasets import (
    Cop4GeoLabelledDataset,
    MaskingMode,
    MaskingStrategy,
)
import random

# Load class mappings
_data_dir = Path(__file__).parent / "data"
with open(_data_dir / "class_mappings.json") as f:
    CLASS_MAPPINGS = json.load(f)


def prepare_training_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_timesteps: int = 12,
    timestep_freq: str = "month",
    augment: bool = True,
    time_explicit: bool = False,
    task_type: Literal["binary", "multiclass"] = "binary",
    num_outputs: int = 1,
    classes_list: Optional[List[str]] = None,
    masking_strategy_train: MaskingStrategy = MaskingStrategy(MaskingMode.NONE),
    masking_strategy_val: MaskingStrategy = MaskingStrategy(MaskingMode.NONE),
    label_jitter=0,
    label_window=0,
) -> Tuple[Cop4GeoLabelledDataset, Cop4GeoLabelledDataset, Cop4GeoLabelledDataset]:
    """
    Prepare training, validation, and test datasets for model training.

    This function creates Cop4GeoLabelledDataset instances from provided dataframes.

    Parameters
    ----------
    train_df : pd.DataFrame
        DataFrame containing training data.
    val_df : pd.DataFrame
        DataFrame containing validation data.
    test_df : pd.DataFrame
        DataFrame containing test data.
    num_timesteps : int, default=12
        Number of timesteps to use for each sample.
    timestep_freq : str, default="month"
        Frequency of timesteps. Can be "month" or "dekad".
    augment : bool, default=True
        Whether to apply data augmentation to the training dataset.
    time_explicit : bool, default=False
        Whether to use explicit time features.
    task_type : Literal["binary", "multiclass"], default="binary"
        Type of classification task.
    num_outputs : int, default=1
        Number of output classes.
    classes_list : Optional[List[str]], default=None
        List of class names. If None, an empty list is used. Required for multiclass task.
    masking_strategy_train : MaskingStrategy, default=askingMode.NONE
        Masking strategy for training dataset.
    masking_strategy_val : MaskingStrategy, default=MaskingMode.NONE
        Masking strategy for validation and test datasets.
    label_jitter : int, default=0
        Jittering true position of label(s). If 0, no jittering is applied.
    label_window : int, default=0
        Expanding true label in the neighboring window. If 0, no windowing is applied.

    Returns
    -------
    Tuple[Cop4GeoLabelledDataset, Cop4GeoLabelledDataset, Cop4GeoLabelledDataset]
        Tuple containing training, validation, and test datasets.
    """
    train_ds = Cop4GeoLabelledDataset(
        train_df,
        num_timesteps=num_timesteps,
        timestep_freq=timestep_freq,
        task_type=task_type,
        num_outputs=num_outputs,
        time_explicit=time_explicit,
        classes_list=classes_list if classes_list is not None else [],
        augment=augment,
        masking_strategy=masking_strategy_train,
        label_jitter=label_jitter,
        label_window=label_window,
    )
    val_ds = Cop4GeoLabelledDataset(
        val_df,
        num_timesteps=num_timesteps,
        timestep_freq=timestep_freq,
        task_type=task_type,
        num_outputs=num_outputs,
        time_explicit=time_explicit,
        classes_list=classes_list if classes_list is not None else [],
        augment=False,  # No augmentation for validation
        masking_strategy=masking_strategy_val,
        label_jitter=0,  # No jittering for validation
        label_window=0,  # No windowing for validation
    )
    test_ds = Cop4GeoLabelledDataset(
        test_df,
        num_timesteps=num_timesteps,
        timestep_freq=timestep_freq,
        task_type=task_type,
        num_outputs=num_outputs,
        time_explicit=time_explicit,
        classes_list=classes_list if classes_list is not None else [],
        augment=False,  # No augmentation for testing
        masking_strategy=masking_strategy_val,
        label_jitter=0,  # No jittering for testing
        label_window=0,  # No windowing for testing
    )
    return train_ds, val_ds, test_ds


def evaluate_finetuned_model(
    finetuned_model,
    test_ds: Cop4GeoLabelledDataset,
    num_workers: int,
    batch_size: int,
    time_explicit: bool = False,
    classes_list: Optional[List[str]] = None,
    mask_positions: Optional[Sequence[int]] = None,
    return_uncertainty: bool = False,
):
    """
    Evaluates a fine-tuned Presto model on a test dataset and calculates performance metrics.
    This function runs the provided model through the test dataset and computes classification
    metrics including precision, recall, F1-score, and support for each class.

    Parameters
    ----------
    finetuned_model : PretrainedPrestoWrapper
        The fine-tuned Presto model to evaluate.
    test_ds : Cop4GeoLabelledDataset
        The test dataset containing samples and ground truth labels.
    num_workers : int
        Number of worker processes for the DataLoader.
    batch_size : int
        Batch size for model evaluation.
    time_explicit : bool, default=False
        Whether to handle time-explicit predictions by only evaluating valid timesteps. Defaults to False.
    classes_list : Optional[List[str]], default=None
        List of class names for multiclass classification.
        Used to map numeric indices to class names in the output.
        Defaults to None. Required for multiclass task.

    Returns
    -------
        pd.DataFrame: A DataFrame containing classification metrics (precision, recall, F1-score, support)
                     for each class, with class names as rows and metrics as columns.

    Raises:
    -------
    ValueError : If the task type in the test dataset is not supported (must be 'binary' or 'multiclass').
    """
    from sklearn.metrics import ConfusionMatrixDisplay, classification_report
    from torch.utils.data import DataLoader

    # storage for full distributions if we need entropy
    all_probs_full: list[np.ndarray] = [] if return_uncertainty else []

    if mask_positions is not None:
        # for each mask‐from position, run the full classification_report,
        # tag it with k, then concatenate
        dfs = []
        for k in mask_positions:
            ds_k = Cop4GeoLabelledDataset(
                test_ds.dataframe,
                task_type=cast(Literal["binary", "multiclass"], test_ds.task_type),
                num_outputs=cast(int, test_ds.num_outputs),
                num_timesteps=test_ds.num_timesteps,
                timestep_freq=test_ds.timestep_freq,
                time_explicit=time_explicit,
                classes_list=classes_list or [],
                augment=False,
                masking_strategy=MaskingStrategy(MaskingMode.FIXED, from_position=k),
                label_jitter=0,
                label_window=0,
            )
            df_k, cm, cm_norm = evaluate_finetuned_model(
                finetuned_model,
                ds_k,
                num_workers,
                batch_size,
                time_explicit,
                classes_list,
                mask_positions=None,  # disable recursion
                return_uncertainty=return_uncertainty,
            )
            df_k["masked_ts_from_pos"] = k
            # Get the timestamp for this mask position
            if ds_k.timestep_freq == "month":
                # Get the first sample's timestamps (assume all samples aligned)
                ts = ds_k[0].timestamps
                # k is 1-based, so subtract 1 for index
                month_idx = min(k - 1, ts.shape[0] - 1)
                month_num = int(ts[month_idx, 1])
                import calendar

                month_label = calendar.month_abbr[month_num]
            elif ds_k.timestep_freq == "dekad":
                ts = ds_k[0].timestamps
                month_idx = min(k - 1, ts.shape[0] - 1)
                month_num = int(ts[month_idx, 1])
                day_num = int(ts[month_idx, 0])
                import calendar

                month_label = f"{calendar.month_abbr[month_num]} {day_num:02d}"
            else:
                month_label = "Unknown"

            df_k["masked_ts_month_label"] = month_label
            dfs.append(df_k)
        return pd.concat(dfs, ignore_index=True), None, None

    # Put model in eval mode
    finetuned_model.eval()

    # Construct the dataloader
    val_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,  # keep as False!
        num_workers=num_workers,
    )
    assert isinstance(val_dl.sampler, torch.utils.data.SequentialSampler)

    # Run the model on the test set
    all_probs = []
    all_preds = []
    all_targets = []

    for batch in val_dl:
        with torch.no_grad():
            # batch may already be a Predictors or a dict collated by DataLoader
            if hasattr(batch, "move_predictors_to_device"):
                batch = batch.move_predictors_to_device(device)
            else:
                # rehydrate from dict→Predictors
                batch = Predictors(**batch).move_predictors_to_device(device)

            model_output = finetuned_model(batch)
            targets = batch.label.cpu().numpy().astype(int)

            if test_ds.task_type == "binary":
                probs = torch.sigmoid(model_output).cpu().numpy()
                preds = (probs > 0.5).astype(int)
            elif test_ds.task_type == "multiclass":
                probs_all = (
                    torch.softmax(model_output, dim=-1).cpu().numpy()
                )  # shape (B,T,C)

                preds = np.argmax(probs_all, axis=-1, keepdims=True)
                probs = np.max(probs_all, axis=-1, keepdims=True)

                preds = preds[targets != NODATAVALUE]
                probs = probs[targets != NODATAVALUE]
                probs_all = probs_all[targets[:, -1] != NODATAVALUE, :]
                targets = targets[targets != NODATAVALUE]

                if return_uncertainty:
                    # flatten batch×timesteps into (N,C)
                    if all_probs_full is not None:
                        all_probs_full.append(
                            probs_all.reshape(-1, probs_all.shape[-1])
                        )
            else:
                raise ValueError(f"Unsupported task type: {test_ds.task_type}")

            # Handle time-explicit predictions by filtering to valid timesteps only
            if time_explicit:
                # Create a mask that identifies where targets are valid (not NODATAVALUE)
                if test_ds.task_type == "binary":
                    valid_mask = targets != NODATAVALUE
                else:  # multiclass
                    valid_mask = targets != NODATAVALUE

                # Flatten everything with masks to keep only valid predictions
                for i in range(targets.shape[0]):
                    sample_valid_mask = valid_mask[i].flatten()
                    if np.any(sample_valid_mask):
                        # Only include samples that have at least one valid target
                        sample_probs = probs[i].flatten()[sample_valid_mask]
                        sample_preds = preds[i].flatten()[sample_valid_mask]
                        sample_targets = targets[i].flatten()[sample_valid_mask]

                        all_probs.append(sample_probs)
                        all_preds.append(sample_preds)
                        all_targets.append(sample_targets)
            else:
                # For non-time-explicit, just flatten and add everything
                all_probs.append(probs.flatten())
                all_preds.append(preds.flatten())
                all_targets.append(targets.flatten())

    if time_explicit:
        all_probs = np.concatenate(all_probs) if all_probs else np.array([])
        all_preds = np.concatenate(all_preds) if all_preds else np.array([])
        all_targets = np.concatenate(all_targets) if all_targets else np.array([])
    else:
        all_probs = np.concatenate(all_probs)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

    # Map numeric indices to class names if necessary
    if test_ds.task_type == "multiclass" and classes_list:
        all_targets_classes = np.array(
            [classes_list[x] if x != NODATAVALUE else "unknown" for x in all_targets]
        )
        all_preds_classes = np.array([classes_list[x] for x in all_preds])

        # Remove any "unknown" targets before classification report
        valid_indices = all_targets_classes != "unknown"
        all_targets = list(all_targets_classes[valid_indices])
        all_preds = list(all_preds_classes[valid_indices])
        if len(all_probs) > 0:
            all_probs_array = np.array(all_probs)[valid_indices]
            all_probs = list(all_probs_array)
        else:
            all_probs = []
    elif test_ds.task_type == "binary":
        # For binary classification, convert to class names
        all_targets = ["crop" if x > 0.5 else "not_crop" for x in all_targets]
        all_preds = list(
            np.array(["crop" if x > 0.5 else "not_crop" for x in all_preds])
        )
        classes_to_use = ["not_crop", "crop"]
    else:
        # Just use the classes as is
        classes_to_use = classes_list if classes_list is not None else []

    results = classification_report(
        all_targets,
        all_preds,
        labels=classes_to_use if test_ds.task_type == "binary" else None,
        output_dict=True,
        zero_division=0,
    )

    cm = ConfusionMatrixDisplay.from_predictions(
        all_targets,
        all_preds,
        xticks_rotation="vertical",
        labels=classes_to_use if test_ds.task_type == "binary" else None,
    )
    cm_norm = ConfusionMatrixDisplay.from_predictions(
        all_targets,
        all_preds,
        xticks_rotation="vertical",
        normalize="true",
        labels=classes_to_use if test_ds.task_type == "binary" else None,
    )

    results_df = pd.DataFrame(results).transpose().reset_index()
    results_df.columns = pd.Index(
        ["class", "precision", "recall", "f1-score", "support"]
    )
    # compute average predictive entropy if requested
    if return_uncertainty:
        from scipy.stats import entropy

        pf = (
            np.concatenate(all_probs_full, axis=0) if all_probs_full else np.empty((0,))
        )  # shape (N, C)
        ent = entropy(pf.T) if pf.size > 0 else np.array([])  # length N
        results_df["avg_entropy"] = ent.mean() if ent.size > 0 else np.nan

    return results_df, cm, cm_norm


def load_dataset(
    parquet_files: Sequence[str | Path],
    timestep_freq: Literal["month", "dekad"] = "month",
    finetune_classes: Optional[List[str]] = "CROPLAND",
    class_mappings: Optional[dict] = None,
    debug=False,
):  # type: ignore):
    if isinstance(parquet_files, (str, Path)):
        # If a single file is provided, convert it to a list
        parquet_files = [parquet_files]

    if debug:
        # select 1st file in debug mode
        parquet_files = parquet_files[:1]
        logger.warning("Debug mode is enabled.")

    df = None
    for f in parquet_files:
        logger.info(f"Processing {f}")
        _data = pd.read_parquet(f, engine="fastparquet")
        _data = _data[_data["sample_id"].notnull()]
        _data["ewoc_code"] = _data["ewoc_code"].astype(int)

        for tcol in ["valid_time", "start_time", "end_time", "timestamp"]:
            if tcol in _data.columns:
                _data[tcol] = pd.to_datetime(_data[tcol], utc=True)
                _data[tcol] = _data[tcol].dt.tz_localize(None)

        _data_pivot = process_parquet(_data, freq=timestep_freq)
        _data_pivot.reset_index(inplace=True)
        df = _data_pivot if df is None else pd.concat([df, _data_pivot])

    df = map_classes(df, finetune_classes, class_mappings=CLASS_MAPPINGS)
    return df


def train_test_val_split(
    df: pd.DataFrame,
    group_sample_by: Optional[str] = None,
    uniform_sample_by: Optional[str] = None,
    sampling_frac: float = 0.8,
    nmin_per_class: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into training, validation, and test sets using either group-based or uniform sampling.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame to split.
    group_sample_by : str, optional
        Column name to use for group-based sampling. If provided, the split is performed by randomly selecting groups for train/val/test.
    uniform_sample_by : str, optional
        Column name to use for uniform sampling. If provided, the split is performed by sampling uniformly within each group.
    sampling_frac : float, default=0.8
        Fraction of data (or groups) to use for the training set.
    nmin_per_class : int, default=5
        Minimum number of samples required per group/class for inclusion in the split (used only with uniform_sample_by).
        
    Returns
    -------
    df_train : pandas.DataFrame
        DataFrame containing the training set.
    df_val : pandas.DataFrame
        DataFrame containing the validation set.
    df_test : pandas.DataFrame
        DataFrame containing the test set.
        
    Raises
    ------
    ValueError
        If neither `group_sample_by` nor `uniform_sample_by` is provided.
    """
    
    random.seed(3)
    if group_sample_by is None and uniform_sample_by is None:
        raise ValueError(
            "Either group_sample_by or uniform_sample_by must be provided to split the data."
        )
    elif group_sample_by is not None:
        parentnames = df[group_sample_by].unique()
        parentname_train = random.sample(
            list(parentnames), int(len(parentnames) * sampling_frac)
        )
        df_sample = df.copy()
        df_train = df_sample[df_sample[group_sample_by].isin(parentname_train)]

        # split in val and test
        df_val_test = df_sample[~df_sample[group_sample_by].isin(parentname_train)]
        parentname_val_test = df_val_test[group_sample_by].unique()
        parentname_val = random.sample(
            list(parentname_val_test), int(len(parentname_val_test) * 0.5)
        )
        df_val = df_val_test[df_val_test[group_sample_by].isin(parentname_val)]
        df_test = df_val_test[~df_val_test[group_sample_by].isin(parentname_val)]

    elif uniform_sample_by is not None:
        group_counts = df[uniform_sample_by].value_counts()
        valid_groups = group_counts[group_counts >= nmin_per_class].index
        if len(valid_groups) != len(group_counts):
            logger.warning(
                f"Some groups have less than {nmin_per_class} samples. They will be excluded from the split."
            )
        else:
            logger.info(
                f"All groups have at least {nmin_per_class} samples. Proceeding with the split."
            )
        df_sample = df[df[uniform_sample_by].isin(valid_groups)].reset_index(drop=True)
        df_train = df_sample.groupby(uniform_sample_by).sample(
            frac=sampling_frac, random_state=3
        )
        df_val_test = df_sample[~df_sample.index.isin(df_train.index)]
        df_val = df_val_test.groupby(uniform_sample_by).sample(frac=0.5, random_state=3)
        df_test = df_val_test[~df_val_test.index.isin(df_val.index)]
    else:
        raise ValueError(
            "Either group_sample_by or uniform_sample_by must be provided to split the data."
        )

    logger.info(
        f"Training set size: {len(df_train)}, {len(df_train)/len(df):2f} total dataset"
    )
    logger.info(
        f"Validation set size: {len(df_val)}, {len(df_val)/len(df):2f} total dataset"
    )
    logger.info(
        f"Test set size: {len(df_test)}, {len(df_test)/len(df):2f} total dataset"
    )

    return df_train, df_val, df_test
