"""TensorRT FoundationStereo Disparity & Depth Processor."""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import cv2
import numpy as np

from foundationpose_perception_pipeline.inference.models import STEREO_MODEL, ModelPaths
from foundationpose_perception_pipeline.inference.trt import TRTEngine

DIVISIBILITY = 32
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

logger = logging.getLogger(__name__)


def normalize_for_model(image: np.ndarray) -> np.ndarray:
    """Scale an RGB image to [0,1] and apply ImageNet statistics."""
    return (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


# -----------------------------------------------------------------------------
# FoundationStereo TensorRT Processor
# -----------------------------------------------------------------------------
class FoundationStereoTrtProcessor:
    """Pure TensorRT FoundationStereo inferencer with deterministic model paths."""

    def __init__(self, model_path: Path | str | None = None):
        self.model_path = (
            Path(model_path).expanduser().resolve() if model_path is not None
            else ModelPaths.configured().preferred(STEREO_MODEL)
        )
        self.engine_path = self.model_path
        self.engine = None
        self._input_shape = None
        # ONNX mode waits for the rectified pair. Its real padded shape determines
        # the static profile, instead of guessing a rig-specific height here.
        if self.model_path.suffix.lower() == ".onnx":
            import tensorrt as trt

            logger_trt = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger_trt)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, logger_trt)
            if parser.parse_from_file(str(self.model_path)):
                for i in range(network.num_inputs):
                    inp = network.get_input(i)
                    if inp.name == "left_image":
                        shape = tuple(inp.shape)
                        self.fixed_hw = tuple(shape[-2:]) if len(shape) == 4 and all(dim > 0 for dim in shape) else None
                        break
                else:
                    self.fixed_hw = None
            else:
                self.fixed_hw = None
            return
        self.engine = TRTEngine(self.engine_path)
        shape = tuple(self.engine.engine.get_tensor_shape("left_image"))
        if any(dim < 0 for dim in shape):
            minimum, _, maximum = self.engine.engine.get_tensor_profile_shape("left_image", 0)
            self.fixed_hw = tuple(minimum[-2:]) if tuple(minimum) == tuple(maximum) else None
        else:
            self.fixed_hw = tuple(shape[-2:])

    def infer_disparity(self, left_rgb: np.ndarray, right_rgb: np.ndarray, pre_shift_px: int = 0) -> np.ndarray:
        if left_rgb.shape != right_rgb.shape:
            raise ValueError(f"Rectified pair shape mismatch: {left_rgb.shape} vs {right_rgb.shape}")
        if not 0 <= pre_shift_px < right_rgb.shape[1]:
            raise ValueError(f"pre_shift_px must be within the image width, got {pre_shift_px}")

        if pre_shift_px:
            shifted = np.empty_like(right_rgb)
            shifted[:, pre_shift_px:] = right_rgb[:, : right_rgb.shape[1] - pre_shift_px]
            shifted[:, :pre_shift_px] = right_rgb[:, :1]
            right_rgb = shifted

        orig_h, orig_w = left_rgb.shape[:2]
        pad_h, pad_w = (-orig_h) % DIVISIBILITY, (-orig_w) % DIVISIBILITY
        if pad_h or pad_w:
            left_rgb = cv2.copyMakeBorder(left_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
            right_rgb = cv2.copyMakeBorder(right_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)

        norm_l = normalize_for_model(left_rgb)
        norm_r = normalize_for_model(right_rgb)

        tensor_l = np.ascontiguousarray(np.transpose(norm_l, (2, 0, 1))[None, ...], dtype=np.float32)
        tensor_r = np.ascontiguousarray(np.transpose(norm_r, (2, 0, 1))[None, ...], dtype=np.float32)

        feed = {"left_image": tensor_l, "right_image": tensor_r}
        if self.model_path.suffix.lower() == ".onnx" and self._input_shape != tensor_l.shape:
            if self.engine is not None:
                self.engine.release()
                self.engine = None
            self.engine = TRTEngine(
                self.model_path, input_shapes={name: value.shape for name, value in feed.items()}
            )
            self.engine_path = self.engine.engine_path
            self._input_shape = tensor_l.shape
        out = self.engine.infer(feed)
        disparity = out["disparity"].reshape(left_rgb.shape[:2])[:orig_h, :orig_w]

        if pre_shift_px:
            disparity = disparity + pre_shift_px

        return disparity

    def release(self) -> None:
        if self.engine is not None:
            self.engine.release()
            self.engine = None
        self._input_shape = None


StereoEngine = FoundationStereoTrtProcessor

_LOADED_ENGINES: list[FoundationStereoTrtProcessor] = []


def load_engine(engine_path: Path | str | None = None) -> FoundationStereoTrtProcessor:
    """Load or reuse a FoundationStereo TensorRT processor."""
    path = (
        Path(engine_path).expanduser().resolve() if engine_path is not None
        else ModelPaths.configured().preferred(STEREO_MODEL)
    )
    for eng in _LOADED_ENGINES:
        if eng.model_path == path:
            return eng
    proc = FoundationStereoTrtProcessor(model_path=path)
    _LOADED_ENGINES.append(proc)
    return proc


def release_engines() -> None:
    """Free device memory for all loaded stereo engines."""
    for eng in _LOADED_ENGINES:
        eng.release()
    _LOADED_ENGINES.clear()
    gc.collect()


__all__ = [
    "FoundationStereoTrtProcessor",
    "StereoEngine",
    "load_engine",
    "normalize_for_model",
    "release_engines",
]
