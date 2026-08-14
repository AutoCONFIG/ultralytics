# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy

import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import DetMetrics, SegmentMetrics
from ultralytics.utils.plotting import plot_images


class DetectSegmentMetrics:
    """Expose explicit branch-prefixed metric namespaces and averaged fitness."""

    def __init__(self) -> None:
        """Initialize independent stock metric owners."""
        self.detect = DetMetrics()
        self.segment = SegmentMetrics()

    @property
    def keys(self) -> list[str]:
        """Return stable branch-prefixed metric keys."""
        return [f"metrics/detect/{key.split('/', 1)[-1]}" for key in self.detect.keys] + [
            f"metrics/segment/{key.split('/', 1)[-1]}" for key in self.segment.keys
        ]

    @property
    def fitness(self) -> float:
        """Average independent branch fitness values."""
        return (self.detect.fitness + self.segment.fitness) / 2

    @property
    def results_dict(self) -> dict[str, float]:
        """Return explicit branch namespaces and a scalar fitness."""
        results = {
            **{
                f"metrics/detect/{key.split('/', 1)[-1]}": value
                for key, value in self.detect.results_dict.items()
                if key != "fitness"
            },
            **{
                f"metrics/segment/{key.split('/', 1)[-1]}": value
                for key, value in self.segment.results_dict.items()
                if key != "fitness"
            },
        }
        results["fitness"] = self.fitness
        return results


class DetectSegmentValidator(DetectionValidator):
    """Validate detection and segmentation branches independently after one forward pass."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None) -> None:
        """Initialize stock branch validators behind one lifecycle."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "detect-segment"
        if self.args.single_cls:
            raise ValueError("single_cls is not supported for independent detect-segment class spaces")
        if self.args.save_json or self.args.save_txt:
            raise NotImplementedError("save_json and save_txt are not supported for detect-segment validation")
        detect_args, segment_args = copy(self.args), copy(self.args)
        detect_args.task = "detect"
        segment_args.task = "segment"
        self.detect_validator = DetectionValidator(dataloader, save_dir, detect_args, _callbacks)
        self.segment_validator = SegmentationValidator(dataloader, save_dir, segment_args, _callbacks)
        self.metrics = DetectSegmentMetrics()

    def _sync(self, validator) -> None:
        """Share runtime state with a stock branch validator."""
        for name in ("data", "device", "training", "dataloader", "save_dir", "speed"):
            setattr(validator, name, getattr(self, name))

    def _batch(self, batch, branch: str, image_indices):
        """Adapt supervised paired targets to a compact stock validator batch."""
        adapted = dict(batch)
        target_batch_idx = batch[f"{branch}_batch_idx"]
        target_mask = (target_batch_idx[:, None] == image_indices).any(1)
        target_batch_idx = target_batch_idx[target_mask]
        remapped_batch_idx = torch.empty_like(target_batch_idx)
        for compact_index, image_index in enumerate(image_indices):
            remapped_batch_idx[target_batch_idx == image_index] = compact_index
        adapted["img"] = batch["img"][image_indices]
        adapted["batch_idx"] = remapped_batch_idx
        adapted["cls"] = batch[f"{branch}_cls"][target_mask]
        adapted["bboxes"] = batch[f"{branch}_bboxes"][target_mask]
        for key in ("im_file", "ori_shape", "resized_shape", "ratio_pad"):
            adapted[key] = tuple(batch[key][index] for index in image_indices.tolist())
        if branch == "segment":
            adapted["masks"] = batch["segment_masks"][image_indices] if self.args.overlap_mask else batch["segment_masks"][target_mask]
        return adapted

    def preprocess(self, batch):
        """Move paired tensors to the validation device once."""
        batch = DetectionValidator.preprocess(self, batch)
        batch["segment_masks"] = batch["segment_masks"].float()
        self.segment_validator.batch_imgsz = batch["img"].shape[2:]
        return batch

    def init_metrics(self, model) -> None:
        """Initialize stock metrics with independent model class names."""
        self._sync(self.detect_validator)
        self._sync(self.segment_validator)
        native_model = model.backend.model if hasattr(model, "backend") and hasattr(model.backend, "model") else model
        original_names = native_model.names
        native_model.names = native_model.detect_names
        self.detect_validator.init_metrics(native_model)
        native_model.names = native_model.segment_names
        self.segment_validator.init_metrics(native_model)
        native_model.names = original_names
        self.metrics.detect = self.detect_validator.metrics
        self.metrics.segment = self.segment_validator.metrics

    def postprocess(self, preds):
        """Run independent stock NMS and mask reconstruction."""
        if len(preds) == 3:
            return self.detect_validator.postprocess(preds[0]), self.segment_validator.postprocess((preds[1], preds[2]))
        return self.detect_validator.postprocess(preds[0]), self.segment_validator.postprocess(preds[1])

    def update_metrics(self, preds, batch) -> None:
        """Update each branch against only its supervised image subset."""
        for branch, validator, branch_preds in (
            ("detect", self.detect_validator, preds[0]),
            ("segment", self.segment_validator, preds[1]),
        ):
            image_indices = batch[f"{branch}_supervised"].nonzero(as_tuple=False).flatten()
            if image_indices.numel():
                validator.update_metrics(
                    [branch_preds[index] for index in image_indices.tolist()], self._batch(batch, branch, image_indices)
                )
        self.seen = self.detect_validator.seen

    def gather_stats(self) -> None:
        """Gather both branch metric states in distributed validation."""
        self.detect_validator.gather_stats()
        self.segment_validator.gather_stats()

    def get_stats(self) -> dict[str, float]:
        """Process both stock metric owners and return explicit namespaces."""
        for validator in (self.detect_validator, self.segment_validator):
            validator.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        return self.metrics.results_dict

    def finalize_metrics(self) -> None:
        """Finalize both stock metric owners."""
        for validator in (self.detect_validator, self.segment_validator):
            self._sync(validator)
            validator.finalize_metrics()
        self.metrics.speed = self.speed

    def print_results(self) -> None:
        """Print each branch with its independent names and counts."""
        self.detect_validator.print_results()
        self.segment_validator.print_results()

    def _filter_supervised(self, batch, branch):
        """Return only images supervised for *branch*, compacting batch indices for plotting.

        Supervision means the corresponding label file exists; empty label files remain valid negative samples.
        Images belonging only to the other branch are removed from the branch visualization entirely.
        """
        supervised = batch[f"{branch}_supervised"]
        image_idx = supervised.nonzero(as_tuple=False).flatten()
        if not len(image_idx):
            return None

        src_idx = batch[f"{branch}_batch_idx"].long()
        index_map = torch.full((len(supervised),), -1, dtype=torch.long, device=src_idx.device)
        index_map[image_idx] = torch.arange(len(image_idx), device=src_idx.device)
        label_mask = index_map[src_idx] >= 0
        labels = {
            "img": batch["img"][image_idx],
            "cls": batch[f"{branch}_cls"][label_mask],
            "bboxes": batch[f"{branch}_bboxes"][label_mask],
            "batch_idx": index_map[src_idx[label_mask]],
        }
        if branch == "segment" and "segment_masks" in batch:
            labels["masks"] = (
                batch["segment_masks"][image_idx]
                if self.args.overlap_mask
                else batch["segment_masks"][label_mask]
            )
        paths = [batch["im_file"][int(i)] for i in image_idx]
        return labels, paths

    def plot_val_samples(self, batch, ni) -> None:
        """Plot validation samples per branch, only images supervised for that branch."""
        for branch in ("detect", "segment"):
            filtered = self._filter_supervised(batch, branch)
            if filtered is None:
                continue
            labels, paths = filtered
            plot_images(
                labels=labels,
                paths=paths,
                fname=self.save_dir / f"val_batch{branch}{ni}_labels.jpg",
                names=self.data[f"{branch}_names"],
                on_plot=self.on_plot,
            )

    def plot_predictions(self, batch, preds, ni) -> None:
        """Plot branch predictions only on images supervised for that branch."""
        for branch, branch_preds in (("detect", preds[0]), ("segment", preds[1])):
            if not branch_preds:
                continue
            image_idx = batch[f"{branch}_supervised"].nonzero(as_tuple=False).flatten()
            if not len(image_idx):
                continue
            names = self.data[f"{branch}_names"]
            max_det = self.args.max_det
            selected_preds = [branch_preds[int(i)] for i in image_idx]
            for compact_idx, pred in enumerate(selected_preds):
                pred["batch_idx"] = torch.ones_like(pred["conf"]) * compact_idx
            keys = selected_preds[0].keys()
            batched_preds = {k: torch.cat([pred[k][:max_det] for pred in selected_preds], dim=0) for k in keys}
            batched_preds["bboxes"] = ops.xyxy2xywh(batched_preds["bboxes"])
            if branch == "segment":
                batched_preds["masks"] = torch.as_tensor(batched_preds["masks"], dtype=torch.uint8).cpu()
            plot_images(
                images=batch["img"][image_idx],
                labels=batched_preds,
                paths=[batch["im_file"][int(i)] for i in image_idx],
                fname=self.save_dir / f"val_batch{branch}{ni}_pred.jpg",
                names=names,
                on_plot=self.on_plot,
            )
