# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.dataset import DetectSegmentDataset
from ultralytics.data.utils import check_detect_segment_dataset


def _write_dataset(root: Path, detect: str | None, segment: str | None, images: int = 1) -> Path:
    image_dir = root / "images"
    image_dir.mkdir(parents=True)
    index = root / "train.txt"
    image_files = []
    for i in range(images):
        image_file = image_dir / f"sample{i}.jpg"
        cv2.imwrite(str(image_file), np.zeros((20, 40, 3), dtype=np.uint8))
        image_files.append(str(image_file))
        if detect is not None:
            (image_dir / f"sample{i}_det.txt").write_text(detect, encoding="utf-8")
        if segment is not None:
            (image_dir / f"sample{i}_seg.txt").write_text(segment, encoding="utf-8")
    index.write_text("\n".join(image_files), encoding="utf-8")
    yaml_file = root / "data.yaml"
    yaml_file.write_text(
        "\n".join(
            (
                f"path: {root.as_posix()}",
                "train: train.txt",
                "val: train.txt",
                "detect_names: [vehicle]",
                "segment_names: [road]",
            )
        ),
        encoding="utf-8",
    )
    return yaml_file


def _dataset(yaml_file: Path, augment: bool = False) -> DetectSegmentDataset:
    data = check_detect_segment_dataset(str(yaml_file))
    assert data["detect_names"] == {0: "vehicle"}
    assert data["segment_names"] == {0: "road"}
    assert data["detect_nc"] == data["segment_nc"] == 1
    hyp = deepcopy(DEFAULT_CFG)
    disabled = (
        "mosaic",
        "mixup",
        "cutmix",
        "copy_paste",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "fliplr",
        "flipud",
    )
    for name in disabled:
        setattr(hyp, name, 0.0)
    return DetectSegmentDataset(
        img_path=data["train"],
        imgsz=64,
        batch_size=2,
        augment=augment,
        hyp=hyp,
        data=data,
        task="detect-segment",
        split="train",
    )


@pytest.mark.parametrize(
    ("detect", "segment", "detect_count", "segment_count"),
    (
        ("0 0.5 0.5 0.5 0.5\n", "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n", 1, 1),
        ("0 0.5 0.5 0.5 0.5\n", None, 1, 0),
        (None, "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n", 0, 1),
    ),
)
def test_paired_dataset_supports_independent_and_empty_targets(
    tmp_path: Path, detect: str | None, segment: str | None, detect_count: int, segment_count: int
):
    dataset = _dataset(_write_dataset(tmp_path, detect, segment))

    sample = dataset[0]

    assert sample["detect_cls"].shape == (detect_count, 1)
    assert sample["detect_bboxes"].shape == (detect_count, 4)
    assert sample["detect_batch_idx"].shape == (detect_count,)
    assert sample["detect_supervised"] is (detect is not None)
    assert sample["segment_cls"].shape == (segment_count, 1)
    assert sample["segment_bboxes"].shape == (segment_count, 4)
    assert sample["segment_batch_idx"].shape == (segment_count,)
    assert sample["segment_supervised"] is (segment is not None)
    assert sample["segment_masks"].shape[0] == 1
    if detect_count and segment_count:
        assert sample["detect_cls"].item() == sample["segment_cls"].item() == 0
        assert torch.allclose(sample["detect_bboxes"], sample["segment_bboxes"])


def test_paired_dataset_distinguishes_missing_from_empty_branch_labels(tmp_path: Path):
    missing = _dataset(_write_dataset(tmp_path / "missing", None, ""))
    empty = _dataset(_write_dataset(tmp_path / "empty", "", None))

    missing_sample = missing[0]
    empty_sample = empty[0]

    assert missing_sample["detect_supervised"] is False
    assert missing_sample["segment_supervised"] is True
    assert empty_sample["detect_supervised"] is True
    assert empty_sample["segment_supervised"] is False


def test_paired_dataset_rejects_images_without_either_label(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="at least one _det.txt or _seg.txt"):
        _dataset(_write_dataset(tmp_path, None, None))


def test_paired_dataset_uses_colocated_suffix_labels(tmp_path: Path):
    yaml_file = _write_dataset(tmp_path, None, "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n")

    sample = _dataset(yaml_file)[0]

    assert (tmp_path / "images" / "sample0_seg.txt").is_file()
    assert sample["segment_supervised"] is True
    assert sample["segment_cls"].shape == (1, 1)


def test_paired_collation_offsets_each_target_stream(tmp_path: Path):
    dataset = _dataset(
        _write_dataset(
            tmp_path,
            "0 0.5 0.5 0.5 0.5\n",
            "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n",
            images=2,
        )
    )

    batch = dataset.collate_fn([dataset[0], dataset[1]])

    assert batch["detect_batch_idx"].tolist() == [0.0, 1.0]
    assert batch["segment_batch_idx"].tolist() == [0.0, 1.0]
    assert batch["detect_supervised"].tolist() == [True, True]
    assert batch["segment_supervised"].tolist() == [True, True]
    assert batch["detect_bboxes"].shape == batch["segment_bboxes"].shape == (2, 4)
    assert batch["segment_masks"].shape[0] == 2


def test_build_routes_only_detect_segment_task_to_paired_dataset(tmp_path: Path):
    data = check_detect_segment_dataset(str(_write_dataset(tmp_path, None, None)))
    cfg = deepcopy(DEFAULT_CFG)
    cfg.task = "detect-segment"
    cfg.imgsz = 64

    dataset = build_yolo_dataset(cfg, data["train"], batch=1, data=data, mode="val")

    assert isinstance(dataset, DetectSegmentDataset)


def test_paired_dataset_rejects_malformed_labels(tmp_path: Path):
    yaml_file = _write_dataset(tmp_path, "0 0.5 0.5\n", None)

    with pytest.raises(ValueError, match="corrupt image/label"):
        _dataset(yaml_file)


def test_paired_dataset_rejects_box_labels_in_segment_stream(tmp_path: Path):
    yaml_file = _write_dataset(tmp_path, None, "0 0.5 0.5 0.5 0.5\n")

    with pytest.raises(ValueError, match="require polygons"):
        _dataset(yaml_file)


@pytest.mark.parametrize("setting", ("mosaic", "mixup", "cutmix", "copy_paste"))
def test_paired_dataset_rejects_unsafe_augmentation(tmp_path: Path, setting: str):
    data = check_detect_segment_dataset(str(_write_dataset(tmp_path, None, None)))
    hyp = deepcopy(DEFAULT_CFG)
    disabled = (
        "mosaic",
        "mixup",
        "cutmix",
        "copy_paste",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "fliplr",
        "flipud",
    )
    for name in disabled:
        setattr(hyp, name, 0.0)
    setattr(hyp, setting, 0.1)

    with pytest.raises(ValueError, match=setting):
        DetectSegmentDataset(
            img_path=data["train"],
            imgsz=64,
            batch_size=1,
            augment=True,
            hyp=hyp,
            data=data,
            task="detect-segment",
            split="train",
        )


def test_paired_dataset_supports_stock_training_config(tmp_path: Path):
    dataset = _dataset(
        _write_dataset(
            tmp_path,
            "0 0.5 0.5 0.5 0.5\n",
            "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n",
        ),
        augment=True,
    )

    sample = dataset[0]

    assert sample["detect_cls"].shape == (1, 1)
    assert sample["segment_cls"].shape == (1, 1)


def test_paired_dataset_preserves_nested_paths_from_configured_image_root(tmp_path: Path):
    image_root = tmp_path / "images"
    image_file = image_root / "region" / "sample.jpg"
    detect_file = image_root / "region" / "sample_det.txt"
    segment_file = image_root / "region" / "sample_seg.txt"
    image_file.parent.mkdir(parents=True)
    cv2.imwrite(str(image_file), np.zeros((20, 40, 3), dtype=np.uint8))
    detect_file.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    segment_file.write_text("0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n", encoding="utf-8")
    yaml_file = tmp_path / "data.yaml"
    yaml_file.write_text(
        "\n".join(
            (
                f"path: {tmp_path.as_posix()}",
                "train: images",
                "val: images",
                "detect_names: [vehicle]",
                "segment_names: [road]",
            )
        ),
        encoding="utf-8",
    )

    dataset = _dataset(yaml_file)

    assert dataset.labels[0]["im_file"] == str(image_file)
    assert dataset.labels[0]["detect_cls"].shape == (1, 1)
    assert dataset.labels[0]["segment_cls"].shape == (1, 1)


def test_paired_dataset_handles_list_image_sources(tmp_path: Path):
    roots = [tmp_path / "images_a", tmp_path / "images_b"]
    for index, image_root in enumerate(roots):
        image_file = image_root / "nested" / f"sample{index}.jpg"
        image_file.parent.mkdir(parents=True)
        cv2.imwrite(str(image_file), np.zeros((20, 40, 3), dtype=np.uint8))
        for suffix, label in (
            ("_det", "0 0.5 0.5 0.5 0.5\n"),
            ("_seg", "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n"),
        ):
            label_file = image_file.with_name(f"sample{index}{suffix}.txt")
            label_file.write_text(label, encoding="utf-8")
    yaml_file = tmp_path / "data.yaml"
    yaml_file.write_text(
        "\n".join(
            (
                f"path: {tmp_path.as_posix()}",
                "train: [images_a, images_b]",
                "val: [images_a, images_b]",
                "detect_names: [vehicle]",
                "segment_names: [road]",
            )
        ),
        encoding="utf-8",
    )

    dataset = _dataset(yaml_file)

    assert len(dataset) == 2
    assert all(label["detect_cls"].shape == (1, 1) for label in dataset.labels)
    assert all(label["segment_cls"].shape == (1, 1) for label in dataset.labels)


def test_paired_dataset_maps_colocated_label_suffix(tmp_path: Path):
    dataset = DetectSegmentDataset.__new__(DetectSegmentDataset)
    image_file = tmp_path / "images" / "subset" / "sample.jpg"

    label_paths = dataset._label_paths([str(image_file)], "_det.txt")

    assert label_paths == [str(tmp_path / "images" / "subset" / "sample_det.txt")]
