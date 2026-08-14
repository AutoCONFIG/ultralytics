# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import pytest
import torch

from ultralytics.models.yolo.detect_segment.val import DetectSegmentValidator
from ultralytics.nn import DetectionSegmentationModel
from ultralytics.nn.modules import Segment26
from ultralytics.utils import ops


@pytest.mark.parametrize(
    ("config", "strides"),
    (
        ("yolo26n-detect-segment.yaml", (8.0, 16.0, 32.0)),
        ("yolo26n-p2-detect-segment.yaml", (4.0, 8.0, 16.0, 32.0)),
        ("yolo26n-p6-detect-segment.yaml", (8.0, 16.0, 32.0, 64.0)),
    ),
)
def test_yolo26_detect_segment_preserves_native_head_contract(config: str, strides: tuple[float, ...]):
    # Given a regular, P2, or P6 YOLO26 composite configuration
    # When the model is built
    model = DetectionSegmentationModel(config, verbose=False)
    head = model.model[-1]

    # Then both branches retain the native YOLO26 end-to-end head contract
    assert model.end2end
    assert head.detect.end2end and head.segment.end2end
    assert head.detect.reg_max == head.segment.reg_max == 1
    assert isinstance(head.segment, Segment26)
    assert tuple(model.stride.tolist()) == strides


def test_yolo26_detect_segment_training_returns_both_assignment_paths():
    # Given a YOLO26 composite model in training mode
    model = DetectionSegmentationModel("yolo26n-detect-segment.yaml", verbose=False).train()

    # When one image is forwarded
    detect_output, segment_output = model(torch.zeros(1, 3, 64, 64))

    # Then both branches expose one-to-many and one-to-one predictions
    assert set(detect_output) == {"one2many", "one2one"}
    assert set(segment_output) == {"one2many", "one2one"}
    assert "proto" in segment_output["one2many"]
    assert "proto" in segment_output["one2one"]


def test_yolo26_detect_segment_loss_backpropagates_through_both_paths():
    # Given one image with detection and segmentation supervision
    model = DetectionSegmentationModel(
        "yolo26n-detect-segment.yaml", detect_nc=1, segment_nc=1, verbose=False
    )
    model.args = type(
        "Args", (), {"box": 7.5, "cls": 0.5, "dfl": 1.5, "overlap_mask": True, "epochs": 100}
    )()
    batch = {
        "img": torch.zeros(1, 3, 64, 64),
        "detect_batch_idx": torch.tensor([0.0]),
        "detect_cls": torch.tensor([[0.0]]),
        "detect_bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
        "detect_supervised": torch.tensor([True]),
        "segment_batch_idx": torch.tensor([0.0]),
        "segment_cls": torch.tensor([[0.0]]),
        "segment_bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
        "segment_supervised": torch.tensor([True]),
        "segment_masks": torch.ones(1, 16, 16),
        "segment_sem_masks": torch.zeros(1, 16, 16),
    }

    # When the joint loss is backpropagated
    loss, _ = model(batch)
    loss.sum().backward()

    # Then all one-to-many and one-to-one branch parameters remain graph-connected
    assert torch.isfinite(loss).all()
    assert loss.abs().sum() > 0
    for child in (model.model[-1].detect, model.model[-1].segment):
        assert all(parameter.grad is not None for parameter in child.parameters() if parameter.requires_grad)


def test_yolo26_p2_segment_validation_handles_high_resolution_prototypes():
    model = DetectionSegmentationModel(
        "yolo26n-p2-detect-segment.yaml", detect_nc=1, segment_nc=1, verbose=False
    ).eval()
    validator = DetectSegmentValidator(
        args={
            "task": "detect-segment",
            "imgsz": 64,
            "single_cls": False,
            "save_json": False,
            "save_txt": False,
            "conf": 0.0,
            "max_det": 10,
            "overlap_mask": True,
            "plots": False,
        }
    )
    validator.device = torch.device("cpu")
    validator.segment_validator.nc = 1
    validator.segment_validator.end2end = True
    process_shapes = []

    def process_mask(protos, masks_in, bboxes, shape):
        process_shapes.append(tuple(shape))
        return ops.process_mask(protos, masks_in, bboxes, shape)

    validator.segment_validator.process = process_mask
    validator.segment_validator.seen = 0
    batch = validator.preprocess(
        {
            "img": torch.zeros(1, 3, 64, 64, dtype=torch.uint8),
            "ori_shape": ((64, 64),),
            "resized_shape": ((64, 64),),
            "ratio_pad": (((1.0, 1.0), (0.0, 0.0)),),
            "im_file": ("image.jpg",),
            "detect_supervised": torch.tensor([False]),
            "detect_batch_idx": torch.empty(0),
            "detect_cls": torch.empty(0, 1),
            "detect_bboxes": torch.empty(0, 4),
            "segment_supervised": torch.tensor([True]),
            "segment_batch_idx": torch.tensor([0.0]),
            "segment_cls": torch.tensor([[0.0]]),
            "segment_bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
            "segment_masks": torch.ones(1, 16, 16, dtype=torch.uint8),
        }
    )

    with torch.no_grad():
        outputs = model(batch["img"])
    predictions = validator.segment_validator.postprocess(outputs[1])
    segment_batch = validator._batch(batch, "segment", torch.tensor([0]))

    assert predictions[0]["masks"].shape[-2:] == (32, 32)
    assert process_shapes == [(64, 64)]
    validator.segment_validator.update_metrics(predictions, segment_batch)
