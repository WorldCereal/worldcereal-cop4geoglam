from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
from loguru import logger
from torch.utils.data import WeightedRandomSampler
from worldcereal.train.datasets import WorldCerealDataset

from prometheo.predictors import (
    DEM_BANDS,
    METEO_BANDS,
    NODATAVALUE,
    S1_BANDS,
    S2_BANDS,
    Predictors,
)


def get_class_weights(
    labels: np.ndarray[Any],
    method: str = "balanced",  # 'balanced', 'log', or 'none'
    clip_range: Optional[tuple] = None,  # e.g. (0.2, 10.0)
    normalize: bool = True,
) -> Dict[int, float]:
    """
    Compute class weights for classification tasks.

    Args:
        labels: list of integer class labels.
        method: 'balanced' (scikit-learn style), or 'log' (log-scaled), or 'none'.
        clip_range: tuple (min, max) to clip weights.
        normalize: whether to rescale weights to mean = 1.

    Returns:
        class_weights_dict: dict mapping class index → weight
    """
    counts = Counter(labels)
    classes = sorted(counts.keys())
    total_samples = sum(counts.values())
    num_classes = len(classes)
    freq = np.array([counts[c] for c in classes], dtype=np.float32)

    if method == "balanced":
        weights = total_samples / (num_classes * freq)
    elif method == "log":
        inv_freq = 1.0 / freq
        weights = np.log1p(inv_freq / np.mean(inv_freq))
    elif method == "none":
        weights = np.ones_like(freq)
    else:
        raise ValueError(f"Unknown method: {method}")

    if clip_range:
        logger.info(f"Clipping weights to range {clip_range}")
        weights = np.clip(weights, clip_range[0], clip_range[1])

    if normalize:
        logger.info("Renormalizing weights to mean = 1")
        weights = weights / weights.mean()

    return dict(zip(classes, weights))


class MaskingMode(str, Enum):
    NONE = "none"
    FIXED = "fixed"
    RANDOM = "random"


@dataclass
class MaskingStrategy:
    mode: MaskingMode
    from_position: Optional[int] = None

    def __post_init__(self):
        if (
            self.mode in {MaskingMode.FIXED, MaskingMode.RANDOM}
            and self.from_position is None
        ):
            raise ValueError(f"'from_position' must be set for mode={self.mode}")



class Cop4GeoDataset(WorldCerealDataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        num_timesteps: int = 12,
        timestep_freq: str = "month",
        task_type: Literal["ssl", "binary", "multiclass"] = "ssl",
        num_outputs: Optional[int] = None,
        augment: bool = False,
        masking_strategy: MaskingStrategy = MaskingStrategy(MaskingMode.NONE),
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
        masking_strategy: MaskingStrategy, optional
            masking strategy to use, by default MaskingMode.NONE.
            If set to FIXED or RANDOM, the from_position must be set.
        """

        super().__init__(
            dataframe, num_timesteps, timestep_freq, task_type, num_outputs, augment
        )

        # masking parameters
        self.masking_strategy = masking_strategy

        if masking_strategy.mode == MaskingMode.FIXED:
            logger.info(
                f"masking enabled: masking from position {masking_strategy.from_position}"
            )
        if masking_strategy.mode == MaskingMode.RANDOM:
            logger.info(
                f"Random mask position enabled: will randomly mask from positions {masking_strategy.from_position} to {num_timesteps - 1}"
            )

    # def _get_center_point(
    #     self, available_timesteps, valid_position, augment, min_edge_buffer
    # ):
    #     """Helper method to decide on the center point based on which to
    #     extract the timesteps."""

    #     # ADAPTATION FOR FRENCH IN-SEASON POC:
    #     # VALID_POSITION IS 1ST OF JUNE -> SHIFT TO FIRST OF APRIL
    #     # AS DEFAULT CENTER POINT
    #     if self.timestep_freq == "month":
    #         valid_position = valid_position - 2  # months
    #     else:
    #         valid_position = valid_position - 6  # dekads

    #     return super()._get_center_point(
    #         available_timesteps, valid_position, augment, min_edge_buffer
    #     )

    @staticmethod
    def sample_mask_position(min_pos: int, max_pos: int, alpha=1.5, beta=2.5):
        """Samples from a Beta distribution skewed toward 0
        alpha < beta → skew left (early cutoffs)
        alpha = beta = 1 → uniform
        alpha > beta → skew right (late cutoffs — not what we want)
        """
        r = np.random.beta(alpha, beta)
        scaled = int(min_pos + r * (max_pos - min_pos))
        return min(max(scaled, min_pos), max_pos)

    def get_inputs(self, row_d: Dict, timestep_positions: List[int]) -> dict:
        # Get latlons
        latlon = np.array([row_d["lat"], row_d["lon"]], dtype=np.float32)

        # Get timestamps belonging to each timestep
        timestamps = self._get_timestamps(row_d, timestep_positions)

        # Initialize inputs
        s1, s2, meteo, dem = self.initialize_inputs()

        # Determine masking position for this sample
        if self.masking_strategy.mode == MaskingMode.FIXED:
            mask_pos = self.masking_strategy.from_position
        elif self.masking_strategy.mode == MaskingMode.RANDOM:
            # Random mask position
            assert self.masking_strategy.from_position is not None
            max_pos = min(self.num_timesteps - 1, len(timestep_positions) - 1)
            mask_pos = self.sample_mask_position(
                self.masking_strategy.from_position, max_pos + 1
            )
        else:
            mask_pos = None

        # Fill inputs
        for src_attr, dst_atr in self.BAND_MAPPING.items():
            keys = [src_attr.format(t) for t in timestep_positions]
            values = np.array([float(row_d[key]) for key in keys], dtype=np.float32)
            idx_valid = values != NODATAVALUE

            if mask_pos is not None:
                # Create in-range mask for positions >= mask_pos
                in_range_mask = np.arange(self.num_timesteps) >= mask_pos

                # Apply the  mask
                values[in_range_mask] = NODATAVALUE

                # Update valid indices based on the combined mask
                idx_valid = idx_valid & ~in_range_mask

            if dst_atr in S2_BANDS:
                s2[..., S2_BANDS.index(dst_atr)] = values
            elif dst_atr in S1_BANDS:
                # convert to dB
                idx_valid = idx_valid & (values > 0)
                values[idx_valid] = 20 * np.log10(values[idx_valid]) - 83
                s1[..., S1_BANDS.index(dst_atr)] = values
            elif dst_atr == "precipitation":
                # scaling, and AgERA5 is in mm, prometheo convention expects m
                values[idx_valid] = values[idx_valid] / (100 * 1000.0)
                meteo[..., METEO_BANDS.index(dst_atr)] = values
            elif dst_atr == "temperature":
                # remove scaling
                values[idx_valid] = values[idx_valid] / 100
                meteo[..., METEO_BANDS.index(dst_atr)] = values
            elif dst_atr in DEM_BANDS:
                values = values[0]  # dem is not temporal
                dem[..., DEM_BANDS.index(dst_atr)] = values
            else:
                raise ValueError(f"Unknown band {dst_atr}")
        return dict(
            s1=s1, s2=s2, meteo=meteo, dem=dem, latlon=latlon, timestamps=timestamps
        )
        

class Cop4GeoLabelledDataset(Cop4GeoDataset):
    def __init__(
        self,
        dataframe,
        task_type: Literal["binary", "multiclass"] = "binary",
        num_outputs: int = 1,
        classes_list: Union[np.ndarray, List[str]] = [],
        time_explicit: bool = False,
        augment: bool = False,
        masking_strategy: MaskingStrategy = MaskingStrategy(MaskingMode.NONE),
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
        time_explicit : bool, optional
            if True, labels respect the full temporal dimension
            to have temporally explicit outputs, by default False
        masking_strategy: MaskingStrategy, optional
            masking strategy to use, by default MaskingMode.NONE.
        label_jitter : int, optional
            ± timesteps to jitter true label pos, for time_explicit only, by default 0.
            Only used if `time_explicit` is True.
        label_window : int, optional
            ± timesteps to expand around label pos (true or moved), for time_explicit only, by default 0.
            Only used if `time_explicit` is True.
        """
        assert task_type in ["binary", "multiclass"], (
            f"Invalid task type `{task_type}` for labelled dataset"
        )

        super().__init__(
            dataframe,
            task_type=task_type,
            num_outputs=num_outputs,
            augment=augment,
            masking_strategy=masking_strategy,
            **kwargs,
        )
        self.classes_list = classes_list
        self.time_explicit = time_explicit
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
            (1, 1, tsteps, 1),
            fill_value=NODATAVALUE,
            dtype=np.int32,
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
                # one global jitter shift
                if self.label_jitter > 0:
                    shift = np.random.randint(-self.label_jitter, self.label_jitter + 1)
                    seq = [int(np.clip(p + shift, 0, T - 1)) for p in seq]
                # one contiguous window around the min→max of seq
                if self.label_window > 0:
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