import importlib.resources
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from loguru import logger
from prometheo.finetune import Hyperparams
from prometheo.finetune import _setup as _prometheo_setup
from prometheo.predictors import Predictors
from prometheo.utils import DEFAULT_SEED, device, seed_everything
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from worldcereal.train.data import (
    map_classes,
    process_parquet,
    remove_small_classes,
    split_df,
)

from worldcereal_cop4geoglam import data
from worldcereal_cop4geoglam.datasets import Cop4GeoLabelledDataset


def get_class_mappings(country: str = "kenya") -> Dict:
    """Method to get the WorldCereal class mappings for downstream task.

    Returns
    -------
    Dict
        the resulting dictionary with the class mappings
    """

    file_path = importlib.resources.files(data).joinpath(
        f"{country}/class_mappings_{country}.json"
    )
    with importlib.resources.as_file(file_path) as actual_file_path:
        if not actual_file_path.exists():
            raise ValueError(
                f"Class mappings file `{file_path}` for country `{country}` does not exist."
            )
    with file_path.open("r") as f:
        CLASS_MAPPINGS = json.load(f)

    return CLASS_MAPPINGS

def get_training_dfs_from_parquet(
    parquet_files: Union[Union[Path, str], List[Union[Path, str]]],
    timestep_freq: Literal["month", "dekad"] = "month",
    finetune_classes: str = "CROPLAND2",
    use_class_membership: bool = False,
    class_mappings: Dict[str, Dict[str, str]] = get_class_mappings(),
    val_samples_file: Optional[Union[Path, str]] = None,
    test_samples_file: Optional[Union[Path, str]] = None,
    debug: bool = False,

) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare training, validation, and test DataFrames from parquet files for presto model fine-tuning.
    This function reads parquet files containing time series data, processes them into a wide format,
    maps the classes according to the specified fine-tuning target, and splits the data into train,
    validation, and test sets.
    Parameters
    ----------
    parquet_files : List[Union[Path, str]]
        List of local paths to parquet files.
    timestep_freq : str, default="month"
        Frequency of timesteps. Can be "month" or "dekad".
    finetune_classes (str):
        The set of fine-tuning classes to use from CLASS_MAPPINGS.
        This should be one of the keys in CLASS_MAPPINGS.
        Most popular maps: "LANDCOVER14", "CROPTYPE9", "CROPTYPE0", "CROPLAND2".
        Defaults to "CROPLAND2".
    class_mappings (dict, optional):
            Dictionary containing the mapping of original class codes to new class labels.
    val_samples_file : Optional[Union[Path, str]], default=None
        Path to a CSV file containing sample IDs for controlled validation set selection.
        If provided, the test set will be constructed using these sample IDs.
        If None, a random train/test split will be performed.
    debug : bool, default=False
        If True, a maximum of one file will be processed for quick testing.
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple containing three DataFrames:
        - train_df: DataFrame with training samples
        - val_df: DataFrame with validation samples
        - test_df: DataFrame with test samples
    """
    logger.info("Reading dataset")
    if isinstance(parquet_files, (str, Path)):
        # If a single file is provided, convert it to a list
        parquet_files = [parquet_files]
    if debug:
        # select 1st file in debug mode
        parquet_files = parquet_files[:1]
        logger.warning("Debug mode is enabled.")
    df = pd.DataFrame()
    sample_memberships = pd.DataFrame(columns=['sample_id', 'membership'])
    for f in parquet_files:
        logger.info(f"Processing {f}")
        _data = pd.read_parquet(f, engine="fastparquet")
        _data = _data[_data["sample_id"].notnull()]
        _data["ewoc_code"] = _data["ewoc_code"].astype(int)
        for tcol in ["valid_time", "start_time", "end_time", "timestamp"]:
            if tcol in _data.columns:
                _data[tcol] = pd.to_datetime(_data[tcol], utc=True)
                _data[tcol] = _data[tcol].dt.tz_localize(None)
        if use_class_membership:
            _data = _data[_data["membership"].notnull()]
            sample_memberships_ = _data[['sample_id', 'membership']]
            sample_memberships_ = sample_memberships_.drop_duplicates(subset=['sample_id'])
            sample_memberships = pd.concat([sample_memberships, sample_memberships_])
            _data = _data.drop(columns=["membership"])
        _data_pivot = process_parquet(_data, freq=timestep_freq)
        _data_pivot.reset_index(inplace=True)
        df = _data_pivot if df is None else pd.concat([df, _data_pivot])
    if use_class_membership:
        sample_memberships = sample_memberships.drop_duplicates(subset=['sample_id']).reset_index(drop=True)
        df = df.merge(sample_memberships, on='sample_id', how='left')
        df.rename(columns={'membership':'finetune_class'}, inplace=True)
    else:
        df = map_classes(df, finetune_classes, class_mappings=class_mappings)

    # Don't apply small classes filtering in case of fuzzy labelling
    if len(df.finetune_class.iloc[0]) == 1:
        # Remove classes with too few samples for stratification
        df = remove_small_classes(df, min_samples=10)
    if test_samples_file is not None:
        logger.info(
            f"Controlled `train/val` vs `test` split based on: {test_samples_file}"
        )
        test_samples_df = pd.read_csv(test_samples_file)
        trainval_df, test_df = split_df(
            df, val_sample_ids=test_samples_df.sample_id.tolist()
        )
    else:
        logger.info("Random `train/val` vs `test` split ...")
        # train_df, test_df = split_df(df, val_size=0.2)
        # TO DO: add possibility of per-class stratification to original split_df function
        trainval_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["finetune_class"]
        )
    # train_df, val_df = split_df(train_df, val_size=0.2)
    # Remove classes with too few samples for stratification, now on trainval_df
    trainval_df = remove_small_classes(trainval_df, min_samples=5)
    if val_samples_file is not None:
        logger.info(f"Controlled `train` vs `val` split based on: {val_samples_file}")
        val_samples_df = pd.read_csv(val_samples_file)
        train_df, val_df = split_df(
            trainval_df, val_sample_ids=val_samples_df.sample_id.tolist()
        )
    else:
        logger.info("Random `train` vs `val` split ...")
        train_df, val_df = train_test_split(
            trainval_df,
            test_size=0.2,
            random_state=42,
            stratify=trainval_df["finetune_class"],
        )
    if test_samples_file:
        # With controlled test set it's possible that either
        # the test set has unique classes not present in training
        # So we need to remove those classes in its totality
        # Detect multi-label (list / tuple / ndarray one-hot) vs single-label (scalar)
        sample_val = train_df["finetune_class"].iloc[0]
        is_multilabel = isinstance(sample_val, (list, tuple, np.ndarray))
        def extract_present_classes(df):
            if df.empty:
                return set()
            if not is_multilabel:
                return set(df["finetune_class"].unique())
            present = set()
            for row in df["finetune_class"]:
                # row is expected a one-hot like sequence
                for idx, v in enumerate(row):
                    if v > 0:
                        present.add(idx)
            return present
        train_classes = extract_present_classes(train_df)
        val_classes = extract_present_classes(val_df)
        test_classes = extract_present_classes(test_df)
        nontrainval_classes = test_classes - (train_classes | val_classes)
        if nontrainval_classes:
            if is_multilabel:
                # Keep only samples whose positive labels are all within train/val classes
                keep_mask = []
                for row in test_df["finetune_class"]:
                    row_classes = {i for i, v in enumerate(row) if v > 0}
                    # Drop if any class is unseen (intersection not empty)
                    keep_mask.append(len(row_classes & nontrainval_classes) == 0)
                before = len(test_df)
                test_df = test_df[keep_mask]
                removed = before - len(test_df)
            else:
                before = len(test_df)
                test_df = test_df[~test_df["finetune_class"].isin(nontrainval_classes)]
                removed = before - len(test_df)
            logger.warning(
                "Removed classes from test set because they do not occur in train/val: "
                f"{sorted(nontrainval_classes)} (samples removed: {removed})"
            )
    return train_df, val_df, test_df

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
    fuzzy_targets: bool = False,
    # masking_strategy_train: MaskingStrategy = MaskingStrategy(MaskingMode.NONE),
    # masking_strategy_val: MaskingStrategy = MaskingStrategy(MaskingMode.NONE),
    label_jitter: int = 0,
    label_window: int = 0,
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
    fuzzy_targets : bool, default=False
        If True, the `finetune_class` column in the dataframe is expected to contain
        soft/fuzzy labels (list/array of class membership probabilities) instead of hard labels.
        Only used if `task_type` is "multiclass".
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
    Tuple[InSeasonLabelledDataset, InSeasonLabelledDataset, InSeasonLabelledDataset]
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
        fuzzy_targets=fuzzy_targets,
        augment=augment,
        # masking_strategy=masking_strategy_train,
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
        fuzzy_targets=fuzzy_targets,
        augment=False,  # No augmentation for validation
        # masking_strategy=masking_strategy_val,
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
        fuzzy_targets=fuzzy_targets,
        augment=False,  # No augmentation for testing
        # masking_strategy=masking_strategy_val,
        label_jitter=0,  # No jittering for testing
        label_window=0,  # No windowing for testing
    )
    return train_ds, val_ds, test_ds


def fuzzy_confusion_matrix_soft(y_true_soft, y_pred_soft, normalize=False):
    """
    Compute a fully fuzzy confusion matrix where both true and predicted
    labels are soft (probabilistic).

    Parameters
    ----------
    y_true_soft : array-like of shape (n_samples, n_classes)
        Soft true label matrix (e.g. one-hot or probabilistic ground truth).
    y_pred_soft : array-like of shape (n_samples, n_classes)
        Predicted probability matrix (e.g. from model.predict_proba()).
    normalize : bool, optional (default=False)
        If True, normalize each row to sum to 1.

    Returns
    -------
    matrix : ndarray of shape (n_classes, n_classes)
        Fuzzy confusion matrix summing probabilities over all samples.
    """
    y_true_soft = np.asarray(y_true_soft)
    y_pred_soft = np.asarray(y_pred_soft)

    # Sanity checks
    assert y_true_soft.shape == y_pred_soft.shape, (
        "y_true_soft and y_pred_soft must have the same shape (n_samples, n_classes)."
    )

    # Core computation: matrix multiplication
    C = y_true_soft.T @ y_pred_soft

    if normalize:
        C = C / C.sum(axis=1, keepdims=True)

    return C


def evaluate_finetuned_model(
    finetuned_model,
    test_ds: Cop4GeoLabelledDataset,
    num_workers: int,
    batch_size: int,
    time_explicit: bool = False,
    classes_list: Optional[List[str]] = None,
):
    """
    Evaluates a fine-tuned Presto model on a test dataset and calculates performance metrics.
    This function runs the provided model through the test dataset and computes classification
    metrics including precision, recall, F1-score, and support for each class.

    Parameters
    ----------
    finetuned_model : PretrainedPrestoWrapper
        The fine-tuned Presto model to evaluate.
    test_ds : InSeasonLabelledDataset
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
    from sklearn.metrics import ConfusionMatrixDisplay
    from torch.utils.data import DataLoader

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
    # all_preds = []
    all_targets = []

    for batch in val_dl:
        with torch.no_grad():
            # batch may already be a Predictors or a dict collated by DataLoader
            if isinstance(batch, dict):
                batch = Predictors(**batch)

            model_output = finetuned_model(batch)

            if test_ds.task_type == "binary":
                probs = torch.sigmoid(model_output).cpu().numpy()
                # preds = (probs > 0.5).astype(int) #### COMMENTED TO PASS RUFF CHECKS> TEMPORARY SOLUTION
            elif test_ds.task_type == "multiclass":
                probs = (
                    torch.softmax(model_output, dim=-1)  # Softmax on the logits
                    .squeeze(
                        dim=[1, 2, 3]
                    )  # Remove space/time dimensions (B, C) remains
                    .cpu()
                    .numpy()
                )  # shape (B,C)
                targets = batch.label.float().squeeze(dim=[1, 2, 3]).cpu().numpy()

                # preds = np.argmax(probs_all, axis=-1, keepdims=True)
                # probs = np.max(probs_all, axis=-1, keepdims=True)

                # preds = preds[targets != NODATAVALUE]
                # probs = probs[targets != NODATAVALUE]
                # probs_all = probs_all[(targets != NODATAVALUE)[..., -1], :]
                # targets = targets[targets != NODATAVALUE]
            else:
                raise ValueError(f"Unsupported task type: {test_ds.task_type}")

            # Handle time-explicit predictions by filtering to valid timesteps only
            if time_explicit:
                raise NotImplementedError
                # # Create a mask that identifies where targets are valid (not NODATAVALUE)
                # valid_mask = targets != NODATAVALUE

                # # Flatten everything with masks to keep only valid predictions
                # for i in range(targets.shape[0]):
                #     sample_valid_mask = valid_mask[i].flatten()
                #     if np.any(sample_valid_mask):
                #         # Only include samples that have at least one valid target
                #         sample_probs = probs[i].flatten()[sample_valid_mask]
                #         sample_preds = preds[i].flatten()[sample_valid_mask]
                #         sample_targets = targets[i].flatten()[sample_valid_mask]

                #         all_probs.append(sample_probs)
                #         all_preds.append(sample_preds)
                #         all_targets.append(sample_targets)
            else:
                all_probs.append(probs)
                # all_preds.append(preds)
                all_targets.append(targets)

    if time_explicit:
        raise NotImplementedError
        # all_probs = np.concatenate(all_probs) if all_probs else np.array([])
        # all_preds = np.concatenate(all_preds) if all_preds else np.array([])
        # all_targets = np.concatenate(all_targets) if all_targets else np.array([])
    else:
        all_probs = np.concatenate(all_probs)
        # all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

    # # Map numeric indices to class names if necessary
    # if test_ds.task_type == "multiclass" and classes_list:
    #     all_targets_classes = np.array(
    #         [classes_list[x] if x != NODATAVALUE else "unknown" for x in all_targets]
    #     )
    #     all_preds_classes = np.array([classes_list[x] for x in all_preds])

    #     # Remove any "unknown" targets before classification report
    #     valid_indices = all_targets_classes != "unknown"
    #     all_targets = list(all_targets_classes[valid_indices])
    #     all_preds = list(all_preds_classes[valid_indices])
    #     if len(all_probs) > 0:
    #         all_probs_array = np.array(all_probs)[valid_indices]
    #         all_probs = list(all_probs_array)
    #     else:
    #         all_probs = []
    # elif test_ds.task_type == "binary":
    #     # For binary classification, convert to class names
    #     all_targets = ["crop" if x > 0.5 else "not_crop" for x in all_targets]
    #     all_preds = list(
    #         np.array(["crop" if x > 0.5 else "not_crop" for x in all_preds])
    #     )
    #     classes_to_use = ["not_crop", "crop"]
    # else:
    #     # Just use the classes as is
    #     classes_to_use = classes_list if classes_list is not None else []

    # results = classification_report(
    #     all_targets,
    #     all_preds,
    #     labels=classes_to_use if test_ds.task_type == "binary" else None,
    #     output_dict=True,
    #     zero_division=0,
    # )

    cm = fuzzy_confusion_matrix_soft(all_targets, all_probs, normalize=False)
    cm_norm = fuzzy_confusion_matrix_soft(all_targets, all_probs, normalize=True)

    cm = ConfusionMatrixDisplay(
        cm,
        display_labels=classes_list,
    )
    cm_norm = ConfusionMatrixDisplay(
        cm_norm,
        display_labels=classes_list,
    )

    # cm = ConfusionMatrixDisplay.from_predictions(
    #     all_targets,
    #     all_preds,
    #     xticks_rotation="vertical",
    #     labels=classes_to_use if test_ds.task_type == "binary" else None,
    # )
    # cm_norm = ConfusionMatrixDisplay.from_predictions(
    #     all_targets,
    #     all_preds,
    #     xticks_rotation="vertical",
    #     normalize="true",
    #     labels=classes_to_use if test_ds.task_type == "binary" else None,
    # )

    # results_df = pd.DataFrame(results).transpose().reset_index()
    # results_df.columns = pd.Index(
    #     ["class", "precision", "recall", "f1-score", "support"]
    # )

    return None, cm, cm_norm, all_targets, all_probs


def run_finetuning(
    model: torch.nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    experiment_name: str,
    output_dir: Union[Path, str],
    loss_fn: torch.nn.Module,
    optimizer: Union[torch.optim.Optimizer],
    scheduler: Union[torch.optim.lr_scheduler.LRScheduler],
    hyperparams: Hyperparams = Hyperparams(),
    seed: int = DEFAULT_SEED,
    setup_logging: bool = True,
    freeze_layers: Optional[List[str]] = None,
    unfreeze_epoch: Optional[int] = None,
):
    output_dir = Path(output_dir)
    _prometheo_setup(output_dir, experiment_name, setup_logging)
    seed_everything()

    # Set model path
    finetuned_model_path = output_dir / f"{experiment_name}.pt"
    finetuned_encoder_path = output_dir / f"{experiment_name}_encoder.pt"
    if finetuned_model_path.is_file():
        raise FileExistsError(
            f"Model file {finetuned_model_path} already exists. Choose a different directory or experiment name."
        )

    train_loss = []
    val_loss = []
    best_loss: Optional[float] = None
    best_model_dict = None
    epochs_since_improvement = 0

    # Track layers that were originally frozen
    originally_frozen_layers = set()

    # Freeze specified layers initially
    if freeze_layers:
        for name, param in model.named_parameters():
            if any(layer in name for layer in freeze_layers):
                if not param.requires_grad:
                    originally_frozen_layers.add(name)
                param.requires_grad = False
                logger.info(f"Freezing layer: {name}")

    for epoch in (pbar := tqdm(range(hyperparams.max_epochs), desc="Finetuning")):
        model.train()

        # Unfreezing logic
        if freeze_layers and epoch == unfreeze_epoch:
            for name, param in model.named_parameters():
                if name not in originally_frozen_layers and any(
                    layer in name for layer in freeze_layers
                ):
                    param.requires_grad = True
                    logger.info(f"Unfreezing layer: {name}")

        epoch_train_loss = 0.0

        for batch in tqdm(train_dl, desc="Training", leave=False):
            optimizer.zero_grad()
            preds = model(batch)
            targets = batch.label.to(device)
            # if preds.dim() > 1 and preds.size(-1) > 1:
            #     # multiclass case: targets should be class indices
            #     # predictions are multiclass logits
            #     targets = targets.long().squeeze(axis=-1)
            # else:
            #     # binary or regression case
            #     targets = targets.float()
            targets = targets.float()

            # Compute loss
            # loss = loss_fn(
            #     preds[targets != NODATAVALUE], targets[targets != NODATAVALUE]
            # )
            loss = loss_fn(preds.squeeze(dim=[1, 2, 3]), targets.squeeze(dim=[1, 2, 3]))

            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss.append(epoch_train_loss / len(train_dl))

        model.eval()
        all_preds, all_y = [], []

        for batch in val_dl:
            with torch.no_grad():
                preds = model(batch)
                targets = batch.label.to(device)

                # if preds.dim() > 1 and preds.size(-1) > 1:
                #     # multiclass case: targets should be class indices
                #     # predictions are multiclass logits
                #     targets = targets.long().squeeze(axis=-1)
                # else:
                #     # binary or regression case
                #     targets = targets.float()
                targets = targets.float()

                # preds = preds[targets != NODATAVALUE]
                # targets = targets[targets != NODATAVALUE]
                all_preds.append(preds)
                all_y.append(targets)

        val_preds = torch.cat(all_preds)
        val_targets = torch.cat(all_y)
        current_val_loss = loss_fn(
            val_preds.squeeze(dim=[1, 2, 3]), val_targets.squeeze(dim=[1, 2, 3])
        ).item()
        val_loss.append(current_val_loss)

        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(current_val_loss)
        else:
            scheduler.step()

        if best_loss is None:
            best_loss = val_loss[-1]
            best_model_dict = deepcopy(model.state_dict())
        else:
            if val_loss[-1] < best_loss:
                best_loss = val_loss[-1]
                best_model_dict = deepcopy(model.state_dict())
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= hyperparams.patience:
                    logger.info("Early stopping!")
                    break

        description = (
            f"Epoch {epoch + 1}/{hyperparams.max_epochs} | "
            f"Train Loss: {train_loss[-1]:.4f} | "
            f"Val Loss: {current_val_loss:.4f} | "
            f"Best Loss: {best_loss:.4f}"
        )

        if epochs_since_improvement > 0:
            description += f" (no improvement for {epochs_since_improvement} epochs)"
        else:
            description += " (improved)"

        pbar.set_description(description)
        pbar.set_postfix(lr=scheduler.get_last_lr()[0])
        logger.info(
            f"PROGRESS after Epoch {epoch + 1}/{hyperparams.max_epochs}: {description}"
        )  # Only log to file if console filters on "PROGRESS"

    assert best_model_dict is not None

    model.load_state_dict(best_model_dict)
    model.eval()

    # Save the best model
    torch.save(model.state_dict(), finetuned_model_path)

    # Save just the encoder
    encoder_model = deepcopy(model)
    encoder_model.head = None
    torch.save(encoder_model.state_dict(), finetuned_encoder_path)

    return model
