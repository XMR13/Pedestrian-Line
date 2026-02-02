from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np

from .nms import NMSConfig, nms
from .types import Detection


@dataclass
class YoloPostConfig:
    """
    Konfigturasi untuk YOLO post processing
    """
    conf_threshold: float = 0.25
    iou_threshold: float = 0.4
    max_detections: int = 50
    # If False, skip NMS and only keep top `max_detections` by score.
    apply_nms: bool = True
    # If True, NMS is class-agnostic (current default behavior).
    # If False, runs per-class NMS then merges results by score.
    class_agnostic_nms: bool = True
    # For (C+4, A) layouts, some exports include an objectness row: (C+5, A).
    # Set True/False to force interpretation; None keeps a simple default.
    anchors_has_objectness: Optional[bool] = None
    # Optional list of class IDs to keep; None keeps all.
    class_ids: Optional[Sequence[int]] = None
    # For anchors layout (4 + C, A), box encoding may vary by exporter.
    # "auto" tries to infer among common conventions; override if needed.
    anchors_box_format: Literal["auto", "cxcywh", "x1y1wh", "xyxy"] = "auto"
    # For decoded layout (N, 6) where last column is class_id, exporter may store
    # box coords as cxcywh or x1y1wh instead of xyxy.
    decoded_box_format: Literal["auto", "cxcywh", "x1y1wh", "xyxy"] = "auto"


class YoloPostprocessor:
    """
    Generic post-process untuk YOLO expoerts:

    Layout yang dibantuk (per image):
    - (N, 5 + C): [cx, cy, w, h, obj, class_scores...]
    - (C + 4, anchor) : contoh 84 x 8400 untuk yolov9/v8 (with anchors)
    - Telah didecoding (N, 6) : [x1, y1, x2, y2, score, class_id]

    Input bisa berupa Numpy: output torch harus dikeluarkan dengan torch.detach() dan dikonversikan 
    menjadi numpy

    """

    def __init__(self, cfg: YoloPostConfig):
        self.cfg = cfg

    def process(
        self,
        preds: np.ndarray,
        orig_size: Tuple[int, int],
        pad: Tuple[float, float] = (0.0, 0.0),
        ratio: Tuple[float, float] = (1.0, 1.0),
    ) -> List[Detection]:
        """
        Mengkonversikan raw output model menjadi filter yang telah didteksi dengan
        koordinat gambar yang original.

        Arg:
            preds: Output model untuk single image
            orig_size: (width, height) untuk gambar awal
            pad: (dw, dh) digunakan kektika letterbox (left/top)
            ratio: (rw, rh) scaling digunakan untuk resize
        """

        input_w = float(orig_size[0] * ratio[0] + 2.0 * pad[0])
        input_h = float(orig_size[1] * ratio[1] + 2.0 * pad[1])
        boxes_xyxy, scores, class_ids = self._decode(preds, input_size=(input_w, input_h))
        if boxes_xyxy.size == 0:
            return []

        boxes_xyxy = self._maybe_denormalize_boxes_xyxy(boxes_xyxy, orig_size=orig_size, pad=pad, ratio=ratio)

        # Filter by score
        keep = scores >= self.cfg.conf_threshold
        boxes_xyxy, scores, class_ids = boxes_xyxy[keep], scores[keep], class_ids[keep]
        if boxes_xyxy.size == 0:
            return []

        # Optional class filter
        if self.cfg.class_ids is not None:
            mask = np.isin(class_ids, np.array(self.cfg.class_ids))
            boxes_xyxy, scores, class_ids = boxes_xyxy[mask], scores[mask], class_ids[mask]
            if boxes_xyxy.size == 0:
                return []

        # NMS (optional) / Top-K
        if self.cfg.apply_nms:
            boxes_xyxy, scores, class_ids = self._apply_nms(boxes_xyxy, scores, class_ids)
        else:
            boxes_xyxy, scores, class_ids = self._select_topk(boxes_xyxy, scores, class_ids)

        # Scale boxes back to original image
        boxes_xyxy = self._scale_boxes(boxes_xyxy, orig_size, pad, ratio)

        return [
            Detection(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                score=float(score),
                class_id=int(cls_id),
            )
            for (x1, y1, x2, y2), score, cls_id in zip(boxes_xyxy, scores, class_ids)
        ]

    # ------------------------------------------------------------------ #
    # Helper internal
    # ------------------------------------------------------------------ #
    @staticmethod
    def _looks_like_class_id_vector(values: np.ndarray) -> bool:
        """
        Heuristic to disambiguate decoded (N, 6) outputs from anchors layouts where (4 + C) can equal 6.

        In decoded layouts, the last column/row is `class_id` and should be (near-)integer.
        In anchors layouts, the last column/row is typically a class score and is not integer-like.
        """

        v = np.asarray(values).astype(np.float32, copy=False)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return False

        # Class IDs are discrete integers; allow tiny floating error.
        is_int = np.isclose(v, np.round(v), atol=1e-3)
        frac_int = float(np.mean(is_int))
        if frac_int < 0.98:
            return False

        # Also require the set of IDs to be small-ish.
        # (Scores rarely quantize perfectly; if they do, they'll typically still have many distinct values.)
        unique_ids = np.unique(np.round(v).astype(np.int64))
        return unique_ids.size <= 256

    @staticmethod
    def _xywh_center_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
        cx, cy, w_box, h_box = boxes_xywh.T
        x1 = cx - w_box / 2
        y1 = cy - h_box / 2
        x2 = cx + w_box / 2
        y2 = cy + h_box / 2
        return np.stack([x1, y1, x2, y2], axis=1)

    @staticmethod
    def _xywh_topleft_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
        x1, y1, w_box, h_box = boxes_xywh.T
        x2 = x1 + w_box
        y2 = y1 + h_box
        return np.stack([x1, y1, x2, y2], axis=1)

    @staticmethod
    def _boxes_validity_score(boxes_xyxy: np.ndarray, *, w: float, h: float) -> float:
        if boxes_xyxy.size == 0 or w <= 0.0 or h <= 0.0:
            return 0.0
        x1, y1, x2, y2 = boxes_xyxy.T
        valid = (x2 > x1) & (y2 > y1)
        # Loose bounds to tolerate minor overshoot / decoding noise.
        valid &= (x1 >= -0.2 * w) & (y1 >= -0.2 * h) & (x2 <= 1.2 * w) & (y2 <= 1.2 * h)
        return float(np.mean(valid)) if valid.size else 0.0

    def _decode_anchors_boxes_to_xyxy(
        self,
        boxes_xywh_raw: np.ndarray,
        *,
        input_size: Optional[Tuple[float, float]],
    ) -> np.ndarray:
        fmt = self.cfg.anchors_box_format
        if fmt == "xyxy":
            return np.array(boxes_xywh_raw, copy=True)
        if fmt == "x1y1wh":
            return self._xywh_topleft_to_xyxy(boxes_xywh_raw)
        if fmt == "cxcywh":
            return self._xywh_center_to_xyxy(boxes_xywh_raw)

        # auto
        raw = boxes_xywh_raw
        # Detect normalized scale vs pixel scale for scoring.
        max_val = float(np.nanmax(raw)) if raw.size else 0.0
        min_val = float(np.nanmin(raw)) if raw.size else 0.0
        is_normalized = (-1.0 <= min_val and max_val <= 2.0)

        if is_normalized:
            w = 1.0
            h = 1.0
        elif input_size is not None:
            w = float(input_size[0])
            h = float(input_size[1])
        else:
            # Fallback: infer rough scale from the raw tensor itself.
            w = max(1.0, max_val)
            h = max(1.0, max_val)

        # Candidate: already xyxy?
        cand_xyxy = np.array(raw, copy=True)
        cand_center = self._xywh_center_to_xyxy(raw)
        cand_topleft = self._xywh_topleft_to_xyxy(raw)

        score_xyxy = self._boxes_validity_score(cand_xyxy, w=w, h=h)
        score_center = self._boxes_validity_score(cand_center, w=w, h=h)
        score_topleft = self._boxes_validity_score(cand_topleft, w=w, h=h)

        best = max(score_xyxy, score_center, score_topleft)
        if best == score_topleft and score_topleft > score_center:
            return cand_topleft
        if best == score_xyxy and score_xyxy > score_center:
            return cand_xyxy
        return cand_center

    def _decode_decoded_boxes_to_xyxy(
        self,
        boxes_raw: np.ndarray,
        *,
        input_size: Optional[Tuple[float, float]],
    ) -> np.ndarray:
        fmt = self.cfg.decoded_box_format
        if fmt == "xyxy":
            return np.array(boxes_raw, copy=True)
        if fmt == "x1y1wh":
            return self._xywh_topleft_to_xyxy(boxes_raw)
        if fmt == "cxcywh":
            return self._xywh_center_to_xyxy(boxes_raw)

        # auto
        raw = boxes_raw
        max_val = float(np.nanmax(raw)) if raw.size else 0.0
        min_val = float(np.nanmin(raw)) if raw.size else 0.0
        is_normalized = (-1.0 <= min_val and max_val <= 2.0)

        if is_normalized:
            w = 1.0
            h = 1.0
        elif input_size is not None:
            w = float(input_size[0])
            h = float(input_size[1])
        else:
            w = max(1.0, max_val)
            h = max(1.0, max_val)

        cand_xyxy = np.array(raw, copy=True)
        cand_center = self._xywh_center_to_xyxy(raw)
        cand_topleft = self._xywh_topleft_to_xyxy(raw)

        score_xyxy = self._boxes_validity_score(cand_xyxy, w=w, h=h)
        score_center = self._boxes_validity_score(cand_center, w=w, h=h)
        score_topleft = self._boxes_validity_score(cand_topleft, w=w, h=h)

        best = max(score_xyxy, score_center, score_topleft)
        if best == score_topleft and score_topleft > score_center:
            return cand_topleft
        if best == score_xyxy and score_xyxy > score_center:
            return cand_xyxy
        return cand_center

    def _maybe_denormalize_boxes_xyxy(
        self,
        boxes_xyxy: np.ndarray,
        *,
        orig_size: Tuple[int, int],
        pad: Tuple[float, float],
        ratio: Tuple[float, float],
    ) -> np.ndarray:
        """
        Some exports (notably certain YOLO forks) emit boxes normalized to [0, 1]
        in letterbox input space. If so, scale them back into pixel coordinates
        of the letterboxed input so the usual pad/ratio inverse mapping works.
        """

        if boxes_xyxy.size == 0:
            return boxes_xyxy

        # Heuristic: if coordinates are mostly in a tiny range, assume normalized.
        # Using a loose upper bound keeps this robust to minor overshoot.
        max_val = float(np.nanmax(boxes_xyxy))
        min_val = float(np.nanmin(boxes_xyxy))
        if not (-1.0 <= min_val and max_val <= 2.0):
            return boxes_xyxy

        dw, dh = pad
        rw, rh = ratio
        orig_w, orig_h = orig_size

        input_w = float(orig_w * rw + 2.0 * dw)
        input_h = float(orig_h * rh + 2.0 * dh)
        if input_w <= 0.0 or input_h <= 0.0:
            return boxes_xyxy

        scaled = np.array(boxes_xyxy, copy=True)
        scaled[:, [0, 2]] *= input_w
        scaled[:, [1, 3]] *= input_h
        return scaled

    def _decode(
        self,
        preds: np.ndarray,
        input_size: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decode variasi dari YOLO layout menjadi kotak dengan koordinat xyxy, score, dan class_ids.
        """

        p = np.asarray(preds)
        if p.ndim == 3:
            if p.shape[0] != 1:
                raise ValueError(f"Batch > 1 is not supported (got shape {p.shape}). Pass one image at a time.")
            p = p[0] #menghilangkan axis dim 0 
        p = np.squeeze(p)

        # Support edge case: a single decoded detection shaped (6,) after squeeze.
        if p.ndim == 1 and p.shape[0] == 6:
            p = p[None, :]

        # kasus: telah terdecode (N, 6) atau (6, N) =>
        # [x1, y1, x2, y2, score, class_id]
        #
        # Note: anchors layout can also be (4 + C, A) where (4 + C) may equal 6 (e.g., C=2).
        # Disambiguate by checking whether the last row/col looks like discrete class IDs.
        if p.ndim == 2:
            # Heuristic: decoded (6, N) outputs typically have small N (#detections),
            # while anchors layouts use huge N (e.g., 8400 / 25200). Avoid misclassifying
            # sparse class-score vectors (many zeros) as integer class IDs.
            large_n_threshold = 2048
            if p.shape[1] == 6 and self._looks_like_class_id_vector(p[:, 5]):
                boxes_raw = np.array(p[:, 0:4], copy=True)
                scores = p[:, 4]
                class_ids = p[:, 5].astype(int)
                boxes_xyxy = self._decode_decoded_boxes_to_xyxy(boxes_raw, input_size=input_size)
                return boxes_xyxy, scores, class_ids
            if (
                p.shape[0] == 6
                and p.shape[1] != 6
                and p.shape[1] <= large_n_threshold
                and self._looks_like_class_id_vector(p[5, :])
            ):
                p_t = p.T
                boxes_raw = np.array(p_t[:, 0:4], copy=True)
                scores = p_t[:, 4]
                class_ids = p_t[:, 5].astype(int)
                boxes_xyxy = self._decode_decoded_boxes_to_xyxy(boxes_raw, input_size=input_size)
                return boxes_xyxy, scores, class_ids


        if p.ndim == 2:
            h, w = p.shape

            # Heuristic: YOLO "channels" dimension is usually small (<= ~512) and anchors dimension is large.
            small, large = (h, w) if h <= w else (w, h)
            is_channels_first = h <= w
            looks_like_anchors_layout = small >= 6 and small <= 512 and (large / max(small, 1)) >= 4

            # Kasus : (C+4, A) atau (C+5, A) (channel yang pertama)
            if looks_like_anchors_layout and is_channels_first:
                boxes_xywh = p[0:4, :].T  # (A, 4)
                rest = p[4:, :]  # (C or 1+C, A)

                if self.cfg.anchors_has_objectness is True:
                    if rest.shape[0] < 2:
                        raise ValueError(f"Expected objectness + class scores, got shape {p.shape}.")
                    objectness = rest[0, :]
                    class_scores = rest[1:, :]
                    class_ids = np.argmax(class_scores, axis=0)
                    class_conf = class_scores[class_ids, np.arange(class_scores.shape[1])]
                    scores = objectness * class_conf
                else:
                    # Some exports emit (4 + 2, A) as:
                    # - (score, class_id)  (already-selected class), OR
                    # - (class0_score, class1_score) (per-class scores)
                    #
                    # Disambiguate by checking for an integer-like class_id row.
                    if rest.shape[0] == 2:
                        r0_is_id = self._looks_like_class_id_vector(rest[0, :])
                        r1_is_id = self._looks_like_class_id_vector(rest[1, :])
                        if r0_is_id and not r1_is_id:
                            class_ids = rest[0, :].astype(int)
                            scores = rest[1, :]
                        elif r1_is_id and not r0_is_id:
                            scores = rest[0, :]
                            class_ids = rest[1, :].astype(int)
                        else:
                            class_scores = rest
                            class_ids = np.argmax(class_scores, axis=0)
                            scores = class_scores[class_ids, np.arange(class_scores.shape[1])]
                    else:
                        # Default: treat remaining rows as class scores only.
                        class_scores = rest
                        class_ids = np.argmax(class_scores, axis=0)
                        scores = class_scores[class_ids, np.arange(class_scores.shape[1])]

                boxes_xyxy = self._decode_anchors_boxes_to_xyxy(boxes_xywh, input_size=input_size)

            # jika memiliki seperti ni (N, 5 + C)
            elif p.shape[1] >= 6:
                boxes = p[:, :4]
                objectness = p[:, 4:5]
                class_scores = p[:, 5:]
                class_ids = np.argmax(class_scores, axis=1)
                class_conf = class_scores[np.arange(class_scores.shape[0]), class_ids]
                scores = objectness[:, 0] * class_conf
                boxes_xyxy = self._xywh_center_to_xyxy(boxes)
            else:
                raise ValueError(f"Format yolo tidak dissuport {p.shape}")

        else:
            raise ValueError(f"Unsupported YOLO output shape: {p.shape}")

        return boxes_xyxy, scores, class_ids

    def _scale_boxes(
        self,
        boxes: np.ndarray,
        orig_size: Tuple[int, int],
        pad: Tuple[float, float],
        ratio: Tuple[float, float],
    ) -> np.ndarray:
        """
        Map boxes dari letterboard ke original image.
        """

        dw, dh = pad
        rw, rh = ratio
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / rw
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / rh

        orig_w, orig_h = orig_size
        boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h - 1)
        return boxes

    def _select_topk(self, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if boxes.size == 0:
            return boxes, scores, class_ids
        if scores.shape[0] <= self.cfg.max_detections:
            return boxes, scores, class_ids
        order = np.argsort(scores)[::-1][: self.cfg.max_detections]
        return boxes[order], scores[order], class_ids[order]

    def _apply_nms(self, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        nms_cfg = NMSConfig(iou_threshold=self.cfg.iou_threshold, max_detections=self.cfg.max_detections)

        if self.cfg.class_agnostic_nms:
            keep_idx = nms(boxes, scores, nms_cfg)
            return boxes[keep_idx], scores[keep_idx], class_ids[keep_idx]

        kept: List[int] = []
        for cls in np.unique(class_ids):
            idx = np.where(class_ids == cls)[0]
            if idx.size == 0:
                continue
            keep_local = nms(boxes[idx], scores[idx], nms_cfg)
            kept.extend(idx[keep_local].tolist())

        if not kept:
            return boxes[:0], scores[:0], class_ids[:0]

        kept = np.array(kept, dtype=np.int32)
        order = np.argsort(scores[kept])[::-1]
        kept = kept[order]
        if kept.size > self.cfg.max_detections:
            kept = kept[: self.cfg.max_detections]
        return boxes[kept], scores[kept], class_ids[kept]
