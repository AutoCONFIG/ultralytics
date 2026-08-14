# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from ultralytics.engine.results import DetectSegmentResults, Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, nms, ops


class DetectSegmentPredictor(DetectionPredictor):
    """Postprocess independent detection and segmentation outputs from one model forward."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """Initialize the native detect-segment predictor."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "detect-segment"

    def postprocess(self, preds, img, orig_imgs):
        """Apply independent NMS and mask reconstruction, returning nested stock results."""
        if self.args.classes is not None:
            raise ValueError("classes filtering is not supported for independent detect-segment class spaces")
        if len(preds) == 3:
            detect_output, segment_predictions, prototypes = preds
            segment_output = segment_predictions, prototypes
        else:
            detect_output, segment_output = preds
        detect_predictions = detect_output[0] if isinstance(detect_output, tuple) else detect_output
        segment_predictions, prototypes = segment_output[0] if isinstance(segment_output[0], tuple) else segment_output
        backend_model = getattr(getattr(self.model, "backend", None), "model", None)
        native_model = backend_model if backend_model is not None else self.model
        detect_names = native_model.detect_names
        segment_names = native_model.segment_names
        end2end = getattr(native_model, "end2end", False)
        detect_preds = nms.non_max_suppression(
            detect_predictions,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(detect_names),
            end2end=end2end,
        )
        segment_preds = nms.non_max_suppression(
            segment_predictions,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(segment_names),
            end2end=end2end,
        )
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
        results = []
        for detect_pred, segment_pred, proto, orig_img, img_path in zip(
            detect_preds, segment_preds, prototypes, orig_imgs, self.batch[0]
        ):
            detect_pred[:, :4] = ops.scale_boxes(img.shape[2:], detect_pred[:, :4], orig_img.shape)
            if segment_pred.shape[0] == 0:
                masks = None
            elif self.args.retina_masks:
                segment_pred[:, :4] = ops.scale_boxes(img.shape[2:], segment_pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto, segment_pred[:, 6:], segment_pred[:, :4], orig_img.shape[:2])
            else:
                masks = ops.process_mask(
                    proto, segment_pred[:, 6:], segment_pred[:, :4], img.shape[2:], upsample=True
                )
                segment_pred[:, :4] = ops.scale_boxes(img.shape[2:], segment_pred[:, :4], orig_img.shape)
            results.append(
                DetectSegmentResults(
                    Results(orig_img, path=img_path, names=detect_names, boxes=detect_pred[:, :6]),
                    Results(orig_img, path=img_path, names=segment_names, boxes=segment_pred[:, :6], masks=masks),
                )
            )
        return results
