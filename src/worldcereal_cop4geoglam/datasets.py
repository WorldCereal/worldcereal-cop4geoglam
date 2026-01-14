from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
from loguru import logger
from prometheo.predictors import (
    NODATAVALUE,
    Predictors,
)
from torch.utils.data import WeightedRandomSampler
from worldcereal.train.datasets import (
    SensorMaskingConfig,
    WorldCerealDataset,
    get_class_weights,
)


def get_class_weights_from_soft_targets(
    target_probs: np.ndarray,
    method: str = "balanced",  # 'balanced', 'log', or 'none'
    clip_range: Optional[tuple] = None,
    normalize: bool = True,
) -> Dict[int, float]:
    """
    Compute class weights for soft/fuzzy targets.

    Args:
        target_probs: array of shape (N, C) with per-class membership probabilities.
        method: same options as the original: 'balanced', 'log', or 'none'.
        clip_range: tuple (min, max) to clip weights.
        normalize: whether to rescale weights to mean = 1.

    Returns:
        class_weights_dict: dict mapping class index → weight
    """
    # Effective soft class counts (sum of membership)
    class_mass = target_probs.sum(axis=0)  # shape (C,)
    total_mass = class_mass.sum()
    num_classes = len(class_mass)

    if method == "balanced":
        # identical to sklearn-style formula: N / (C * n_c)
        weights = total_mass / (num_classes * class_mass)
    elif method == "log":
        inv_freq = 1.0 / class_mass
        weights = np.log1p(inv_freq / np.mean(inv_freq))
    elif method == "none":
        weights = np.ones_like(class_mass)
    else:
        raise ValueError(f"Unknown method: {method}")

    if clip_range:
        weights = np.clip(weights, clip_range[0], clip_range[1])

    if normalize:
        weights = weights / weights.mean()

    return {c: float(w) for c, w in enumerate(weights)}


class Cop4GeoDataset(WorldCerealDataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        num_timesteps: int = 12,
        timestep_freq: str = "month",
        task_type: Literal["ssl", "binary", "multiclass"] = "ssl",
        num_outputs: Optional[int] = None,
        augment: bool = False,
        masking_config: Optional[SensorMaskingConfig] = None,
    ):
        """WorldCereal base dataset. This dataset is typically used for
        self-supervised learning.

        Parameters
        ----------
        dataframe : pd.DataFrame
            input dataframe containing the data
        num_timesteps : int, optional
            number of timesteps for a sample, by default 12
        timestep_freq : str, optional. Should be one of ['month', 'dekad']
            frequency of the timesteps, by default "month"
        task_type : str, optional. One of ['ssl', 'binary', 'multiclass']
            type of the task, by default self-supervised learning "ssl"
        num_outputs : int, optional
            number of outputs for the task, by default None. If task_type is 'ssl',
            the value of this parameter is ignored.
        augment : bool, optional
            whether to augment the data, by default False
        masking_config : Optional[SensorMaskingConfig], optional
            configuration for sensor masking during training, by default None.
        """

        super().__init__(
            dataframe,
            num_timesteps,
            timestep_freq,
            task_type,
            num_outputs,
            augment,
            masking_config=masking_config,
        )


class Cop4GeoLabelledDataset(Cop4GeoDataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        task_type: Literal["binary", "multiclass"] = "binary",
        num_outputs: int = 1,
        classes_list: Union[np.ndarray, List[str]] = [],
        fuzzy_targets: bool = False,
        time_explicit: bool = False,
        augment: bool = False,
        masking_config: Optional[SensorMaskingConfig] = None,
        label_jitter: int = 0,  # ± timesteps to jitter true label pos, for time_explicit only
        label_window: int = 0,  # ± timesteps to expand around label pos (true or moved), for time_explicit only
        return_sample_id: bool = False,
        **kwargs,
    ):
        """Labelled version of WorldCerealDataset for supervised training.
        Additional arguments are explained below.

        Parameters
        ----------
        num_outputs : int, optional
            number of outputs to supervise training on, by default 1
        classes_list : List, optional
            list of column names in the dataframe containing class labels for multiclass tasks,
            used to extract labels from each row of the dataframe, by default []
        fuzzy_targets : bool, optional
            if True, the `finetune_class` column in the dataframe is expected to contain
            soft/fuzzy labels (list/array of class membership probabilities) instead of hard labels,
            by default False. Only used if `task_type` is "multiclass".
        time_explicit : bool, optional
            if True, labels respect the full temporal dimension
            to have temporally explicit outputs, by default False
        masking_config : Optional[SensorMaskingConfig], optional
            configuration for sensor masking during training, by default None.
        label_jitter : int, optional
            ± timesteps to jitter true label pos, for time_explicit only, by default 0.
            Only used if `time_explicit` is True.
        label_window : int, optional
            ± timesteps to expand around label pos (true or moved), for time_explicit only, by default 0.
            Only used if `time_explicit` is True.
        """
        assert task_type in [
            "binary",
            "multiclass",
        ], f"Invalid task type `{task_type}` for labelled dataset"

        super().__init__(
            dataframe,
            task_type=task_type,
            num_outputs=num_outputs,
            augment=augment,
            masking_config=masking_config,
            **kwargs,
        )
        self.classes_list = classes_list
        self.fuzzy_targets = fuzzy_targets
        if task_type == "binary":
            assert not fuzzy_targets, "fuzzy_targets not supported for binary task"
        self.time_explicit = time_explicit
        if time_explicit and fuzzy_targets:
            raise NotImplementedError(
                "time_explicit=True not yet implemented for fuzzy_targets=True"
            )
        self.label_jitter = label_jitter
        self.label_window = label_window
        self.return_sample_id = return_sample_id

        if self.return_sample_id and "sample_id" not in self.dataframe.columns:
            raise ValueError(
                "`return_sample_id` is True, but 'sample_id' column not found in dataframe."
            )

    def __getitem__(self, idx):
        row = pd.Series.to_dict(self.dataframe.iloc[idx, :])
        timestep_positions, valid_position = self.get_timestep_positions(row)
        inputs = self.get_inputs(row, timestep_positions)
        label = self.get_label(
            row,
            task_type=self.task_type,
            classes_list=self.classes_list,
            valid_position=valid_position - timestep_positions[0],
        )

        predictors = Predictors(
            **inputs, label=label
        )  # <<< Create Predictors object first

        if self.return_sample_id:
            sample_id = row["sample_id"]
            return predictors, sample_id
        else:
            return predictors

    def initialize_label(self):
        tsteps = self.num_timesteps if self.time_explicit else 1
        label = np.full(
            (1, 1, tsteps, self.num_outputs if self.fuzzy_targets else 1),
            fill_value=NODATAVALUE
            if not self.fuzzy_targets
            else np.nan,  # Fuzzy targets cannot work with NODATAVALUE
            dtype=np.float32 if self.fuzzy_targets else np.int32,
        )  # [H, W, T or 1, 1]

        return label

    def get_label(
        self,
        row_d: Dict,
        task_type: str = "binary",
        classes_list: Optional[List] = None,
        valid_position: Optional[
            Union[int, Sequence[int]]
        ] = None,  # TO DO: this can also be a list of positions
    ) -> np.ndarray:
        """Get the label for the given row. Label is a 2D array based on
        the number of timesteps and number of outputs. If time_explicit is False,
        the number of timesteps will be set to 1.

        Parameters
        ----------
        row_d : Dict
            input row as a dictionary
        task_type : str, optional
            task type to infer labels from, by default "binary"
        classes_list : Optional[List], optional
            list of column names in the dataframe containing class labels for multiclass tasks,
            must be provided if task_type is "multiclass", by default None
        valid_position : int, optional
            the ‘true’ timestep index where the label lives, by default None.
            If provided and `time_explicit` is True,
            only the label at the corresponding timestep will be
            set while other timesteps will be set to NODATAVALUE.
            We’ll optionally jitter it and/or expand it into a small time‐window.

        Returns
        -------
        np.ndarray
            label array
        """

        label = self.initialize_label()
        T = self.num_timesteps

        # 1) determine base position (single int) or all-positions if not time_explicit
        base_idxs: List[int]
        if not self.time_explicit:
            base_idxs = [0]
        else:
            if valid_position is None:
                # putting label at every timestep
                base_idxs = list(range(T))
            elif isinstance(valid_position, (list, tuple, np.ndarray)):
                # bring into a flat Python list of ints
                if isinstance(valid_position, np.ndarray):
                    seq: List[int] = valid_position.astype(int).tolist()
                else:
                    seq = [int(x) for x in valid_position]
                # Apply either jittering or label_window, but not both
                if self.label_jitter > 0 and self.label_window > 0:
                    apply_jitter = np.random.choice([True, False])
                else:
                    apply_jitter = self.label_jitter > 0

                if apply_jitter:
                    # one global jitter shift
                    shift = np.random.randint(-self.label_jitter, self.label_jitter + 1)
                    seq = [int(np.clip(p + shift, 0, T - 1)) for p in seq]
                elif self.label_window > 0:
                    # one contiguous window around the min→max of seq
                    mn = min(seq)
                    mx = max(seq)
                    start = max(0, mn - self.label_window)
                    end = min(T - 1, mx + self.label_window)
                    base_idxs = list(range(start, end + 1))
                else:
                    base_idxs = seq
            else:
                # apply jitter
                # scalar valid_position must be an int here
                assert isinstance(valid_position, int), (
                    f"Expected single int valid_position, got {type(valid_position)}"
                )
                p = valid_position
                if self.label_jitter > 0:
                    shift = np.random.randint(-self.label_jitter, self.label_jitter + 1)
                    p = int(np.clip(p + shift, 0, T - 1))
                # apply window expansion
                if self.label_window > 0:
                    start = max(0, p - self.label_window)
                    end = min(T - 1, p + self.label_window)
                    base_idxs = list(range(start, end + 1))
                else:
                    base_idxs = [p]

        valid_idx = np.array(base_idxs, dtype=int)

        # 2) set the labels at those indices
        if task_type == "binary":
            label[0, 0, valid_idx, 0] = int(
                not row_d["finetune_class"].startswith("not_")
            )
        elif task_type == "multiclass":
            if not classes_list:
                raise ValueError("classes_list should be provided for multiclass task")
            if self.fuzzy_targets:
                label[0, 0, valid_idx, :] = row_d["finetune_class"]
            else:
                label[0, 0, valid_idx, 0] = classes_list.index(row_d["finetune_class"])

        return label

    def get_balanced_sampler(
        self,
        method: str = "balanced",
        clip_range: Optional[tuple] = None,  # e.g. (0.2, 10.0)
        normalize: bool = True,
        generator: Optional[Any] = None,
        sampling_class: str = "finetune_class",
    ) -> "WeightedRandomSampler":
        """
        Build a WeightedRandomSampler so that rare classes (from `balancing_class`)
        are upsampled and common classes downsampled.
        max_upsample:
            maximum upsampling factor for the rarest class (e.g. 10 means
            no class will be sampled >10× more than its frequency).
        sampling_class:
            column name in the dataframe to use for balancing.
            Default is `finetune_class`, which is the class label
            used in the training. `balancing_class` can be used as well.
        """
        # extract the sampling class (strings or ints)
        bc_vals = self.dataframe[sampling_class].values

        logger.info("Computing class weights ...")

        if type(bc_vals[0]) is list:
            # Have to follow the fuzzy part here
            bc_vals = np.vstack(bc_vals)
            class_weights = get_class_weights_from_soft_targets(
                bc_vals, method, clip_range=clip_range, normalize=normalize
            )
            logger.info(f"Class weights: {class_weights}")

            # Convert dict -> array for broadcasting
            class_weight_vec = np.array(
                [class_weights[c] for c in range(len(class_weights))], dtype=np.float32
            )

            # Each sample's weight = expected class weight under its label distribution
            sample_weights = (
                (bc_vals * class_weight_vec[None, :]).sum(axis=1).astype(np.float32)
            )

        else:
            class_weights = get_class_weights(
                bc_vals, method, clip_range=clip_range, normalize=normalize
            )
            logger.info(f"Class weights: {class_weights}")

            # per‐sample weight
            sample_weights = np.ones_like(bc_vals).astype(np.float32)
            for k, v in class_weights.items():
                sample_weights[bc_vals == k] = v

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        return sampler
