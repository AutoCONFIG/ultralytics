# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path

import numpy as np
import torch

from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionSegmentationModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.plotting import plot_images, plot_labels


class DetectSegmentTrainer(yolo.detect.DetectionTrainer):
    """Train a shared-backbone model with independent detection and segmentation targets."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: dict | None = None):
        """Initialize the native detect-segment trainer."""
        overrides = {} if overrides is None else overrides
        overrides["task"] = "detect-segment"
        if overrides.get("single_cls"):
            raise ValueError("single_cls is not supported for independent detect-segment class spaces")
        super().__init__(cfg, overrides, _callbacks)
        if self.args.single_cls:
            raise ValueError("single_cls is not supported for independent detect-segment class spaces")

    def get_model(self, cfg: dict | str | None = None, weights: str | Path | None = None, verbose: bool = True):
        """Build the composite model with dataset branch class counts."""
        model = DetectionSegmentationModel(
            cfg,
            ch=self.data["channels"],
            detect_nc=self.data["detect_nc"],
            segment_nc=self.data["segment_nc"],
            verbose=verbose and RANK == -1,
        )
        model.detect_names = self.data["detect_names"]
        model.segment_names = self.data["segment_names"]
        model.names = model.detect_names
        if weights:
            model.load(weights)
        return model

    def set_model_attributes(self):
        """Attach independent names and training arguments to the model."""
        self.model.detect_names = self.data["detect_names"]
        self.model.segment_names = self.data["segment_names"]
        self.model.names = self.model.detect_names
        self.model.args = self.args

    def set_class_weights(self):
        """Leave branch classification losses unweighted for the MVP."""

    def auto_batch(self):
        """Estimate batching from the larger paired target stream."""
        train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        max_num_obj = max(
            max(len(label["detect_cls"]), len(label["segment_cls"])) for label in train_dataset.labels
        )
        n = len(train_dataset)
        del train_dataset
        return super(yolo.detect.DetectionTrainer, self).auto_batch(max_num_obj, dataset_size=n)

    def progress_string(self):
        """Return a formatted string of training progress with branch-prefixed losses."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_labels(self):
        """Plot independent detection and segmentation label distributions."""
        for branch, names_key in (("detect", "detect_names"), ("segment", "segment_names")):
            boxes = np.concatenate([lb[f"{branch}_bboxes"] for lb in self.train_loader.dataset.labels], 0)
            cls = np.concatenate([lb[f"{branch}_cls"] for lb in self.train_loader.dataset.labels], 0)
            if len(cls) == 0:
                continue
            plot_labels(
                boxes,
                cls.squeeze(1) if cls.ndim == 2 else cls,
                names=self.data[names_key],
                save_dir=self.save_dir / branch,
                on_plot=self.on_plot,
            )

    def _filter_supervised(self, batch, branch):
        """Return only images supervised for *branch*, with remapped batch_idx."""
        supervised = batch[f"{branch}_supervised"]
        indices = supervised.nonzero(as_tuple=False).flatten()
        if indices.numel() == 0:
            return None
        src_idx = batch[f"{branch}_batch_idx"]
        mask = (src_idx[:, None] == indices).any(1)
        remapped = torch.empty(src_idx.shape[0], dtype=src_idx.dtype, device=src_idx.device)
        for compact, original in enumerate(indices.tolist()):
            remapped[src_idx == original] = compact
        labels = {
            "img": batch["img"][indices],
            "cls": batch[f"{branch}_cls"][mask],
            "bboxes": batch[f"{branch}_bboxes"][mask],
            "batch_idx": remapped[mask],
        }
        if branch == "segment" and "segment_masks" in batch:
            labels["masks"] = batch["segment_masks"][indices]
        paths = [batch["im_file"][i] for i in indices.tolist()]
        return labels, paths

    def plot_training_samples(self, batch, ni) -> None:
        """Plot training samples per branch, only images supervised for that branch."""
        for branch in ("detect", "segment"):
            filtered = self._filter_supervised(batch, branch)
            if filtered is None:
                continue
            labels, paths = filtered
            plot_images(
                labels=labels,
                paths=paths,
                fname=self.save_dir / f"train_batch{branch}{ni}.jpg",
                names=self.data[f"{branch}_names"],
                on_plot=self.on_plot,
            )

    def get_validator(self):
        """Return the native branch-composed validator."""
        return yolo.detect_segment.DetectSegmentValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
