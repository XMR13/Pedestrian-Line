import onnxruntime as ort
import numpy as np
import cv2
import yaml
from pathlib import Path

# ======================== CONFIG ========================
MODEL_PATH = "Models/yolov9-s_v2.onnx"     # <-- change to your .onnx path
DATA_YAML  = "Models/metadata.yaml"         # <-- your YAML (the one you showed)
IMAGE_PATH = "media/tes3.jpeg"        # <-- path to your test image

IMG_SIZE   = 640
CONF_THRES = 0.5
IOU_THRES  = 0.55

# Legacy debug helper for COCO-like models.
# If you are using a custom vehicle-subclass model (truck/trailer/pickup/etc),
# prefer loading names via a YOLO `data.yaml` and update `ALLOWED_CLASS_IDS` accordingly.
ALLOWED_CLASS_IDS = {0, 2, 7, 3}
# ========================================================

def load_class_names(yaml_path):
    p = Path(yaml_path)
    if not p.exists():
        print(f"[WARN] YAML not found at {p}, using built-in COCO names")
        return [
            "person","bicycle","car","motorcycle","airplane","bus","train","truck",
            "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
            "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
            "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
            "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
            "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
            "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
            "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
            "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
            "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
            "hair drier","toothbrush",
        ]
    with p.open("r") as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]
    return names


def letterbox(im, new_shape=640, color=(114, 114, 114)):
    """Resize and pad to square like YOLO."""
    shape = im.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right  = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def xywh2xyxy(x):
    """Convert [cx,cy,w,h] to [x1,y1,x2,y2]."""
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def box_iou(box1, box2):
    # box1: (N,4), box2: (M,4)
    if box1.ndim == 1:
        box1 = box1[None, :]
    if box2.ndim == 1:
        box2 = box2[None, :]
    N = box1.shape[0]
    M = box2.shape[0]
    ious = np.zeros((N, M), dtype=np.float32)
    for i in range(N):
        x11, y11, x12, y12 = box1[i]
        area1 = (x12 - x11) * (y12 - y11)
        for j in range(M):
            x21, y21, x22, y22 = box2[j]
            xx1 = max(x11, x21)
            yy1 = max(y11, y21)
            xx2 = min(x12, x22)
            yy2 = min(y12, y21)
            w = max(0.0, xx2 - xx1)
            h = max(0.0, yy2 - yy1)
            inter = w * h
            area2 = (x22 - x21) * (y22 - y21)
            union = area1 + area2 - inter + 1e-7
            ious[i, j] = inter / union
    return ious


def nms(boxes, scores, iou_thres):
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = box_iou(boxes[i], boxes[idxs[1:]])[0]
        idxs = idxs[1:][ious < iou_thres]
    return keep


def main():
    # ---- load class names ----
    names = load_class_names(DATA_YAML)
    print(f"[INFO] Loaded {len(names)} classes")

    # ---- load image ----
    img0 = cv2.imread(IMAGE_PATH)
    if img0 is None:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
    h0, w0 = img0.shape[:2]

    # ---- preprocess ----
    img, r, (dw, dh) = letterbox(img0, IMG_SIZE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_input = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_input = img_input[None, :, :, :]  # (1,3,640,640)

    # ---- create ONNXRuntime session (GPU if available) ----
    print("[INFO] Available providers:", ort.get_available_providers())
    sess = ort.InferenceSession(
        MODEL_PATH,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    print(f"[INFO] Input name: {inp_name}, output name: {out_name}")

    # ---- run inference ----
    print("[INFO] Running inference...")
    outputs = sess.run([out_name], {inp_name: img_input})[0]  # expect (1,84,8400)
    print("[INFO] Raw output shape:", outputs.shape)

    # reshape to (8400, 84)
    pred = outputs[0].T  # (84,8400) -> (8400,84)

    # YOLOv8/9: 4 box coords + 80 class scores
    boxes = pred[:, :4]         # (8400,4)
    class_scores = pred[:, 4:]  # (8400,80)

    # best class & score per anchor
    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores.max(axis=1)

    # ---- filter by confidence + allowed classes ----
    mask = confidences > CONF_THRES
    if ALLOWED_CLASS_IDS:
        mask &= np.isin(class_ids, list(ALLOWED_CLASS_IDS))

    boxes       = boxes[mask]
    confidences = confidences[mask]
    class_ids   = class_ids[mask]

    print(f"[INFO] Detections after threshold: {len(boxes)}")
    if len(boxes) == 0:
        print("[INFO] No detections above threshold. Try lowering CONF_THRES or allowing more classes.")
        return

    # ---- convert boxes and undo letterbox ----
    boxes_xyxy = xywh2xyxy(boxes)

    # undo padding & scale
    boxes_xyxy[:, [0, 2]] -= dw
    boxes_xyxy[:, [1, 3]] -= dh
    boxes_xyxy[:, [0, 2]] /= r
    boxes_xyxy[:, [1, 3]] /= r

    # clip to original image size
    boxes_xyxy[:, 0::2] = np.clip(boxes_xyxy[:, 0::2], 0, w0)
    boxes_xyxy[:, 1::2] = np.clip(boxes_xyxy[:, 1::2], 0, h0)

    # ---- NMS ----
    keep = nms(boxes_xyxy, confidences, IOU_THRES)
    boxes_xyxy = boxes_xyxy[keep]
    confidences = confidences[keep]
    class_ids = class_ids[keep]

    print(f"[INFO] Detections after NMS: {len(boxes_xyxy)}")

    # ---- draw detections ----
    for box, score, cid in zip(boxes_xyxy, confidences, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        label = names[cid] if cid < len(names) else str(cid)
        text = f"{label} {score:.2f}"
        cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img0, text, (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
        )

    out_path = "media/result_yolov9_gpu1.jpg"
    cv2.imwrite(out_path, img0)
    print(f"[INFO] Saved result to {out_path}")


if __name__ == "__main__":
    main()
