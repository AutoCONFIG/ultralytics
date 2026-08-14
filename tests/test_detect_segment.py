# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ultralytics import YOLO
from ultralytics.cfg import TASK2MODEL, entrypoint
from ultralytics.engine.results import DetectSegmentResults, Results
from ultralytics.models.yolo.detect_segment import DetectSegmentPredictor
from ultralytics.models.yolo.detect_segment.val import DetectSegmentMetrics, DetectSegmentValidator
from ultralytics.nn import DetectionSegmentationModel
from ultralytics.nn.modules import Detect, DetectSegment, Segment
from ultralytics.nn.tasks import DetectionModel, SegmentationModel, guess_model_task, yaml_model_load
from ultralytics.utils.loss import DetectSegmentLoss


CFG = "yolov8n-detect-segment.yaml"
DEFAULT_CFG = "yolo26n-detect-segment.yaml"


def test_detect_segment_model_builds_with_independent_class_counts():
    model = DetectionSegmentationModel(CFG, verbose=False)
    head = model.model[-1]

    assert isinstance(head, DetectSegment)
    assert isinstance(head.detect, Detect)
    assert isinstance(head.segment, Segment)
    assert head.detect.nc == 80
    assert head.segment.nc == 2
    assert torch.equal(head.detect.stride, torch.tensor([8.0, 16.0, 32.0]))
    assert torch.equal(head.segment.stride, head.detect.stride)


def test_detect_segment_default_model_uses_yolo26():
    model = DetectionSegmentationModel(verbose=False)
    head = model.model[-1]

    assert model.yaml["yaml_file"] == DEFAULT_CFG
    assert head.detect.reg_max == 1


def test_detect_segment_task_inference_from_config_path_and_model():
    config = yaml_model_load(CFG)
    model = DetectionSegmentationModel(config, verbose=False)

    assert guess_model_task(config) == "detect-segment"
    assert guess_model_task(CFG) == "detect-segment"
    assert guess_model_task(model) == "detect-segment"


def test_detect_segment_training_preserves_child_contracts_and_shared_features():
    model = DetectionSegmentationModel(CFG, verbose=False).train()
    head = model.model[-1]
    child_inputs = []
    hooks = [
        head.detect.register_forward_pre_hook(lambda _module, args: child_inputs.append(args[0])),
        head.segment.register_forward_pre_hook(lambda _module, args: child_inputs.append(args[0])),
    ]

    try:
        detect_output, segment_output = model(torch.zeros(1, 3, 64, 64))
    finally:
        for hook in hooks:
            hook.remove()

    assert set(detect_output) == {"boxes", "scores", "feats"}
    assert set(segment_output) == {"boxes", "scores", "feats", "mask_coefficient", "proto"}
    assert detect_output["scores"].shape == (1, 80, 84)
    assert segment_output["scores"].shape == (1, 2, 84)
    assert segment_output["mask_coefficient"].shape == (1, 32, 84)
    assert segment_output["proto"].shape == (1, 32, 16, 16)
    assert child_inputs[0] is child_inputs[1]
    assert all(detect_feature is segment_feature for detect_feature, segment_feature in zip(*child_inputs))


def test_detect_segment_evaluation_preserves_child_contracts():
    model = DetectionSegmentationModel(CFG, verbose=False).eval()

    with torch.no_grad():
        detect_output, segment_output = model(torch.zeros(1, 3, 64, 64))

    detect_predictions, detect_raw = detect_output
    (segment_predictions, prototypes), segment_raw = segment_output
    assert detect_predictions.shape == (1, 84, 84)
    assert segment_predictions.shape == (1, 38, 84)
    assert prototypes.shape == (1, 32, 16, 16)
    assert detect_raw["scores"].shape[1] == 80
    assert segment_raw["scores"].shape[1] == 2


def test_detect_segment_rejects_tta():
    model = DetectionSegmentationModel(CFG, verbose=False).eval()
    with pytest.raises(NotImplementedError, match="augment=True"):
        model.predict(torch.zeros(1, 3, 64, 64), augment=True)


def test_stock_detection_and_segmentation_models_still_build():
    assert isinstance(DetectionModel("yolov8n.yaml", verbose=False).model[-1], Detect)
    assert isinstance(SegmentationModel("yolov8n-seg.yaml", verbose=False).model[-1], Segment)


@pytest.mark.parametrize(("detect", "segment"), ((True, True), (True, False), (False, True), (False, False)))
def test_detect_segment_loss_is_finite_and_graph_connected(detect: bool, segment: bool):
    model = DetectionSegmentationModel(CFG, detect_nc=1, segment_nc=1, verbose=False)
    model.args = type("Args", (), {"box": 7.5, "cls": 0.5, "dfl": 1.5, "overlap_mask": True})()
    batch = {
        "img": torch.zeros(1, 3, 64, 64),
        "detect_batch_idx": torch.tensor([0.0]) if detect else torch.empty(0),
        "detect_cls": torch.tensor([[0.0]]) if detect else torch.empty(0, 1),
        "detect_bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]) if detect else torch.empty(0, 4),
        "detect_supervised": torch.tensor([detect]),
        "segment_batch_idx": torch.tensor([0.0]) if segment else torch.empty(0),
        "segment_cls": torch.tensor([[0.0]]) if segment else torch.empty(0, 1),
        "segment_bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]) if segment else torch.empty(0, 4),
        "segment_supervised": torch.tensor([segment]),
        "segment_masks": torch.ones(1, 16, 16) if segment else torch.zeros(1, 16, 16),
    }

    loss, items = model(batch)
    loss.sum().backward()

    assert isinstance(model.criterion, DetectSegmentLoss)
    assert loss.shape == (8,)
    assert tuple(items) == model.criterion.loss_names
    assert torch.isfinite(loss).all()
    head = model.model[-1]
    for child in (head.detect, head.segment):
        missing = [name for name, parameter in child.named_parameters() if parameter.requires_grad and parameter.grad is None]
        assert not missing, missing


def test_detect_segment_loss_accepts_eval_prediction_containers():
    model = DetectionSegmentationModel(CFG, detect_nc=1, segment_nc=1, verbose=False).eval()
    model.args = type("Args", (), {"box": 7.5, "cls": 0.5, "dfl": 1.5, "overlap_mask": True})()
    batch = {
        "img": torch.zeros(1, 3, 64, 64),
        "detect_batch_idx": torch.empty(0),
        "detect_cls": torch.empty(0, 1),
        "detect_bboxes": torch.empty(0, 4),
        "detect_supervised": torch.tensor([False]),
        "segment_batch_idx": torch.empty(0),
        "segment_cls": torch.empty(0, 1),
        "segment_bboxes": torch.empty(0, 4),
        "segment_supervised": torch.tensor([False]),
        "segment_masks": torch.zeros(1, 16, 16),
    }
    predictions = model(batch["img"])

    loss, items = model.loss(batch, predictions)

    assert loss.shape == (8,)
    assert tuple(items) == model.criterion.loss_names
    assert torch.isfinite(loss).all()


def test_detect_segment_loss_excludes_unsupervised_images_from_each_branch():
    model = DetectionSegmentationModel(CFG, detect_nc=1, segment_nc=1, verbose=False)
    model.args = type("Args", (), {"box": 7.5, "cls": 0.5, "dfl": 1.5, "overlap_mask": True})()
    batch = {
        "img": torch.zeros(2, 3, 64, 64),
        "detect_batch_idx": torch.tensor([0.0]),
        "detect_cls": torch.tensor([[0.0]]),
        "detect_bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
        "detect_supervised": torch.tensor([True, False]),
        "segment_batch_idx": torch.tensor([1.0]),
        "segment_cls": torch.tensor([[0.0]]),
        "segment_bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
        "segment_supervised": torch.tensor([False, True]),
        "segment_masks": torch.ones(2, 16, 16),
    }

    loss, _ = model(batch)
    loss.sum().backward()

    assert torch.isfinite(loss).all()
    assert all(parameter.grad is not None for parameter in model.model[-1].detect.parameters() if parameter.requires_grad)
    assert all(parameter.grad is not None for parameter in model.model[-1].segment.parameters() if parameter.requires_grad)


def test_detect_segment_cli_default_stays_out_of_released_model_map(monkeypatch):
    selected = {}

    class FakeYOLO:
        task = "detect-segment"

        def __init__(self, model, task=None):
            selected["model"] = model
            selected["task"] = task

        def predict(self, **kwargs):
            selected["predict"] = kwargs

    monkeypatch.setattr("ultralytics.YOLO", FakeYOLO)

    entrypoint("yolo task=detect-segment mode=predict source=image.jpg")

    assert "detect-segment" not in TASK2MODEL
    assert selected["model"] == DEFAULT_CFG
    assert selected["task"] == "detect-segment"


def test_public_yolo_task_map_and_nested_results_contract():
    yolo = YOLO(CFG)
    detect = Results(np.zeros((8, 8, 3), dtype=np.uint8), "image.jpg", {0: "detect"}, boxes=torch.zeros(1, 6))
    segment = Results(
        np.zeros((8, 8, 3), dtype=np.uint8),
        "image.jpg",
        {0: "segment"},
        boxes=torch.zeros(1, 6),
        masks=torch.ones(1, 8, 8),
    )
    result = DetectSegmentResults(detect, segment)

    assert yolo.task == "detect-segment"
    assert yolo.task_map["detect-segment"]["model"] is DetectionSegmentationModel
    assert result.detect.names[0] == "detect"
    assert result.segment.names[0] == "segment"
    assert result.cpu().segment.masks.data.device.type == "cpu"
    assert result.numpy().detect.boxes.data.shape == (1, 6)
    assert set(result.summary()) == {"detect", "segment"}


def test_detect_segment_results_support_predictor_write_lifecycle(tmp_path):
    predictor = DetectSegmentPredictor(overrides={"verbose": True, "save_txt": True})
    predictor.save_dir = tmp_path
    predictor.source_type = SimpleNamespace(stream=False, from_img=False, tensor=False)
    predictor.dataset = SimpleNamespace(mode="image", count=0)
    predictor.results = [
        DetectSegmentResults(
            Results(
                np.zeros((8, 8, 3), dtype=np.uint8),
                "image.jpg",
                {0: "vehicle"},
                boxes=torch.tensor([[1.0, 1.0, 4.0, 4.0, 0.9, 0.0]]),
            ),
            Results(
                np.zeros((8, 8, 3), dtype=np.uint8),
                "image.jpg",
                {0: "road"},
                boxes=torch.tensor([[1.0, 1.0, 4.0, 4.0, 0.8, 0.0]]),
                masks=torch.ones(1, 8, 8),
            ),
        )
    ]
    predictor.results[0].speed = {"preprocess": 0.0, "inference": 1.5, "postprocess": 0.0}

    message = predictor.write_results(0, tmp_path / "image.jpg", torch.zeros(1, 3, 8, 8), [""])

    result = predictor.results[0]
    converted = result.cpu()
    assert "detect:" in message and "segment:" in message
    assert (tmp_path / "labels" / "image_detect.txt").exists()
    assert (tmp_path / "labels" / "image_segment.txt").exists()
    assert converted.speed == result.speed
    assert converted.save_dir == result.save_dir == str(tmp_path)
    assert len(converted.detect) == len(converted.segment) == 1
    with pytest.raises(TypeError):
        result[0]


def test_detect_segment_predictor_preserves_independent_class_zero_names():
    predictor = DetectSegmentPredictor(overrides={"conf": 0.01})
    predictor.model = type(
        "Backend",
        (),
        {
            "backend": type(
                "Native",
                (),
                {"model": type("Model", (), {"detect_names": {0: "vehicle"}, "segment_names": {0: "road"}})()},
            )()
        },
    )()
    predictor.batch = (["image.jpg"],)
    detect = torch.tensor([[[2.0], [2.0], [2.0], [2.0], [0.9]]])
    segment = torch.tensor([[[2.0], [2.0], [2.0], [2.0], [0.9], [1.0]]])
    proto = torch.ones(1, 1, 4, 4)

    result = predictor.postprocess(
        ((detect, {}), (((segment, proto), {}))),
        torch.zeros(1, 3, 8, 8),
        [np.zeros((8, 8, 3), dtype=np.uint8)],
    )[0]

    assert result.detect.names == {0: "vehicle"}
    assert result.segment.names == {0: "road"}
    assert result.detect.boxes.cls.tolist() == [0.0]
    assert result.segment.boxes.cls.tolist() == [0.0]


def test_detect_segment_predictor_rejects_shared_class_filter():
    predictor = DetectSegmentPredictor(overrides={"classes": [0]})

    with pytest.raises(ValueError, match="classes"):
        predictor.postprocess((None, None), None, None)


def test_detect_segment_checkpoint_roundtrip(tmp_path):
    yolo = YOLO(CFG)
    yolo.model.detect_names = {0: "vehicle"}
    yolo.model.segment_names = {0: "road"}
    yolo.model.names = yolo.model.detect_names
    checkpoint = tmp_path / "detect-segment.pt"
    torch.save({"model": yolo.model, "train_args": {"task": "detect-segment"}}, checkpoint)

    loaded = YOLO(checkpoint)

    assert loaded.task == "detect-segment"
    assert loaded.model.detect_names == {0: "vehicle"}
    assert loaded.model.segment_names == {0: "road"}


def test_detect_segment_model_apply_migrates_both_head_caches():
    model = DetectionSegmentationModel(CFG, verbose=False).eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, 64, 64))

    model.half()
    head = model.model[-1]

    assert model.stride.dtype == torch.float16
    assert head.detect.stride is head.segment.stride
    assert head.detect.anchors.dtype == head.segment.anchors.dtype == torch.float16
    assert head.detect.strides.dtype == head.segment.strides.dtype == torch.float16


def test_detect_segment_validator_preprocesses_real_segment_masks_for_child_update(tmp_path):
    validator = DetectSegmentValidator(
        save_dir=tmp_path,
        args={"task": "detect-segment", "imgsz": 64, "single_cls": False, "save_json": False, "save_txt": False},
    )
    validator.device = torch.device("cpu")
    batch = {
        "img": torch.zeros(1, 3, 8, 8, dtype=torch.uint8),
        "ori_shape": ((8, 8),),
        "resized_shape": ((8, 8),),
        "ratio_pad": ((1, 0),),
        "im_file": ("image.jpg",),
        "detect_batch_idx": torch.empty(0),
        "detect_cls": torch.empty(0, 1),
        "detect_bboxes": torch.empty(0, 4),
        "segment_batch_idx": torch.tensor([0.0]),
        "segment_cls": torch.tensor([[0.0]]),
        "segment_bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
        "segment_masks": torch.ones(1, 2, 2, dtype=torch.uint8),
    }

    processed = validator.preprocess(batch)
    segment_batch = validator._batch(processed, "segment", torch.tensor([0]))

    assert processed["segment_masks"].dtype == torch.float32
    assert segment_batch["masks"] is processed["segment_masks"]


def test_detect_segment_validator_routes_only_supervised_images_to_each_child():
    validator = object.__new__(DetectSegmentValidator)
    validator.args = SimpleNamespace(overlap_mask=True)
    calls = {}

    class CaptureValidator:
        seen = 0

        def update_metrics(self, preds, batch):
            calls[self.branch] = (preds, batch)

    detect_validator = CaptureValidator()
    detect_validator.branch = "detect"
    segment_validator = CaptureValidator()
    segment_validator.branch = "segment"
    validator.detect_validator = detect_validator
    validator.segment_validator = segment_validator

    batch = {
        "img": torch.zeros(3, 3, 8, 8),
        "ori_shape": ((8, 8), (9, 9), (10, 10)),
        "resized_shape": ((8, 8), (9, 9), (10, 10)),
        "ratio_pad": ((1, 0), (1, 1), (1, 2)),
        "im_file": ("detect.jpg", "segment.jpg", "both.jpg"),
        "detect_supervised": torch.tensor([True, False, True]),
        "detect_batch_idx": torch.tensor([0.0, 2.0]),
        "detect_cls": torch.tensor([[0.0], [1.0]]),
        "detect_bboxes": torch.tensor([[0.1, 0.1, 0.2, 0.2], [0.2, 0.2, 0.3, 0.3]]),
        "segment_supervised": torch.tensor([False, True, False]),
        "segment_batch_idx": torch.tensor([1.0]),
        "segment_cls": torch.tensor([[0.0]]),
        "segment_bboxes": torch.tensor([[0.3, 0.3, 0.4, 0.4]]),
        "segment_masks": torch.tensor([[[0]], [[1]], [[2]]]),
    }
    preds = (
        [{"bboxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]])}, {"bboxes": torch.tensor([[1.0, 1.0, 2.0, 2.0]])}, {"bboxes": torch.tensor([[2.0, 2.0, 3.0, 3.0]])}],
        [{"bboxes": torch.tensor([[3.0, 3.0, 4.0, 4.0]])}, {"bboxes": torch.tensor([[4.0, 4.0, 5.0, 5.0]])}, {"bboxes": torch.tensor([[5.0, 5.0, 6.0, 6.0]])}],
    )

    validator.update_metrics(preds, batch)

    detect_preds, detect_batch = calls["detect"]
    assert len(detect_preds) == 2
    assert [pred["bboxes"][0, 0].item() for pred in detect_preds] == [0.0, 2.0]
    assert detect_batch["img"].shape[0] == 2
    assert detect_batch["im_file"] == ("detect.jpg", "both.jpg")
    assert detect_batch["ori_shape"] == ((8, 8), (10, 10))
    assert detect_batch["resized_shape"] == ((8, 8), (10, 10))
    assert detect_batch["ratio_pad"] == ((1, 0), (1, 2))
    assert detect_batch["batch_idx"].tolist() == [0.0, 1.0]
    assert detect_batch["cls"].tolist() == [[0.0], [1.0]]

    segment_preds, segment_batch = calls["segment"]
    assert len(segment_preds) == 1
    assert segment_preds[0]["bboxes"][0, 0].item() == 4.0
    assert segment_batch["img"].shape[0] == 1
    assert segment_batch["im_file"] == ("segment.jpg",)
    assert segment_batch["ori_shape"] == ((9, 9),)
    assert segment_batch["resized_shape"] == ((9, 9),)
    assert segment_batch["ratio_pad"] == ((1, 1),)
    assert segment_batch["batch_idx"].tolist() == [0.0]
    assert segment_batch["masks"].tolist() == [[[1]]]


def test_detect_segment_validator_keeps_empty_supervised_images_and_skips_empty_branches():
    validator = object.__new__(DetectSegmentValidator)
    validator.args = SimpleNamespace(overlap_mask=True)
    calls = {}

    class CaptureValidator:
        seen = 4

        def update_metrics(self, preds, batch):
            calls[self.branch] = (preds, batch)

    detect_validator = CaptureValidator()
    detect_validator.branch = "detect"
    segment_validator = CaptureValidator()
    segment_validator.branch = "segment"
    validator.detect_validator = detect_validator
    validator.segment_validator = segment_validator

    batch = {
        "img": torch.zeros(2, 3, 8, 8),
        "ori_shape": ((8, 8), (9, 9)),
        "resized_shape": ((8, 8), (9, 9)),
        "ratio_pad": ((1, 0), (1, 1)),
        "im_file": ("empty-detect.txt.jpg", "missing-detect.txt.jpg"),
        "detect_supervised": torch.tensor([True, False]),
        "detect_batch_idx": torch.empty(0),
        "detect_cls": torch.empty(0, 1),
        "detect_bboxes": torch.empty(0, 4),
        "segment_supervised": torch.tensor([False, False]),
        "segment_batch_idx": torch.empty(0),
        "segment_cls": torch.empty(0, 1),
        "segment_bboxes": torch.empty(0, 4),
        "segment_masks": torch.zeros(2, 1, 1),
    }
    preds = ([{"bboxes": torch.empty(0, 4)}, {"bboxes": torch.empty(0, 4)}], [{}, {}])

    validator.update_metrics(preds, batch)

    detect_preds, detect_batch = calls["detect"]
    assert len(detect_preds) == 1
    assert detect_batch["img"].shape[0] == 1
    assert detect_batch["im_file"] == ("empty-detect.txt.jpg",)
    assert detect_batch["batch_idx"].numel() == 0
    assert detect_batch["cls"].shape == (0, 1)
    assert "segment" not in calls
    assert validator.seen == 4


def test_detect_segment_disables_generic_name_remapping_for_independent_heads():
    source = DetectionSegmentationModel(CFG, detect_nc=2, segment_nc=2, verbose=False)
    target = DetectionSegmentationModel(CFG, detect_nc=2, segment_nc=2, verbose=False)
    source.names = source.detect_names = {0: "car", 1: "person"}
    source.segment_names = {0: "road", 1: "sky"}
    target.names = target.detect_names = {0: "person", 1: "car"}
    target.segment_names = {0: "sky", 1: "road"}
    state = source.state_dict()
    keys = set(state)

    remapped = target._remap_cls_by_names(state, source, verbose=False)

    assert remapped == 0
    assert set(state) == keys


def test_detect_segment_metrics_have_independent_namespaces_and_average_fitness():
    metrics = DetectSegmentMetrics()

    results = metrics.results_dict

    assert "metrics/detect/mAP50-95(B)" in results
    assert "metrics/segment/mAP50-95(M)" in results
    assert results["fitness"] == (metrics.detect.fitness + metrics.segment.fitness) / 2
