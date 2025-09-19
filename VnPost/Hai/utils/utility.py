import math
import time

import cv2
import numpy as np
from ultralytics.engine.results import OBB

""" Các hàm để xử lý yolo detection raw output từ onnx
"""


def preprocess_yolo_input(img: np.ndarray, target_size: int = 640) -> np.ndarray:
    """
    Preprocess input image for YOLO ONNX model.

    Args:
        img (np.ndarray): Original image in BGR format (from cv2.imread)
        target_size (int): Size to resize both width and height to (default: 320)

    Returns:
        np.ndarray: Preprocessed image tensor of shape (1, 3, target_size, target_size)
    """
    img_resized = cv2.resize(img, (target_size, target_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_input = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0  # HWC → CHW
    img_input = np.expand_dims(img_input, axis=0)  # Add batch dim → (1, 3, H, W)
    return img_input


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from center format (x, y, w, h) to corner format (x1, y1, x2, y2).

    Args:
        boxes (np.ndarray): shape (N, 4) in format [x_center, y_center, width, height]

    Returns:
        np.ndarray: shape (N, 4) in format [x1, y1, x2, y2]
    """
    x, y, w, h = boxes.T
    return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=1)


def iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """
    Compute Intersection over Union (IoU) between a box and multiple boxes.

    Args:
        box1 (np.ndarray): shape (1, 4)
        box2 (np.ndarray): shape (N, 4)

    Returns:
        np.ndarray: IoU scores of shape (N,)
    """
    area1 = (box1[0, 2] - box1[0, 0]) * (box1[0, 3] - box1[0, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    inter_x1 = np.maximum(box1[0, 0], box2[:, 0])
    inter_y1 = np.maximum(box1[0, 1], box2[:, 1])
    inter_x2 = np.minimum(box1[0, 2], box2[:, 2])
    inter_y2 = np.minimum(box1[0, 3], box2[:, 3])

    inter_w = np.clip(inter_x2 - inter_x1, a_min=0, a_max=None)
    inter_h = np.clip(inter_y2 - inter_y1, a_min=0, a_max=None)
    inter_area = inter_w * inter_h

    union_area = area1 + area2 - inter_area
    return inter_area / (union_area + 1e-6)


def nms_classless(
    output: np.ndarray, conf_thresh: float = 0.4, iou_thresh: float = 0.5
) -> np.ndarray:
    """
    Apply Non-Maximum Suppression (NMS) without class consideration.

    Args:
        output (np.ndarray): Model raw output of shape (1, 5, num_boxes) in format [x, y, w, h, conf]
        conf_thresh (float): Confidence threshold to filter boxes
        iou_thresh (float): IoU threshold to remove overlapping boxes

    Returns:
        np.ndarray: Filtered boxes after NMS, shape (M, 5) in format [x1, y1, x2, y2, conf]
    """
    output = output.squeeze(0).T  # → (num_boxes, 5)
    boxes = xywh2xyxy(output[:, :4])
    scores = output[:, 4]

    # Filter by confidence
    keep = scores > conf_thresh
    boxes = boxes[keep]
    scores = scores[keep]

    # Sort by confidence
    order = np.argsort(-scores)
    boxes = boxes[order]
    scores = scores[order]

    keep_boxes = []

    while len(boxes) > 0:
        box = boxes[0]
        score = scores[0]
        keep_boxes.append(np.concatenate([box, [score]]))

        if len(boxes) == 1:
            break

        ious = iou(box[np.newaxis, :], boxes[1:])
        keep_idx = ious < iou_thresh
        boxes = boxes[1:][keep_idx]
        scores = scores[1:][keep_idx]

    return np.stack(keep_boxes) if keep_boxes else np.empty((0, 5))


def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = x.copy()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def non_max_suppression_numpy(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    max_det=300,
    max_nms=30000,
    max_time_img=0.05,
    return_idxs=False,
):
    assert prediction.ndim == 3  # [batch, num_channels, num_boxes]

    bs, ch, nb = prediction.shape
    assert bs == 1, "Batch size > 1 not supported in this implementation."

    nc = ch - 5  # 4 bbox + 1 obj conf + rest are class scores
    prediction = np.transpose(prediction, (0, 2, 1))  # [1, num_boxes, num_channels]
    x = prediction[0]

    time_limit = 2.0 + max_time_img
    t_start = time.time()

    boxes = xywh2xyxy(x[:, :4])
    obj_conf = x[:, 4:5]
    cls_conf = x[:, 5 : 5 + nc]
    scores = obj_conf * cls_conf
    cls_idxs = np.argmax(scores, axis=1)
    conf = np.max(scores, axis=1)

    mask = conf > conf_thres
    if not np.any(mask):
        return (
            (np.zeros((0, ch + 1)), np.zeros((0,), dtype=int))
            if return_idxs
            else np.zeros((0, ch + 1))
        )

    x = x[mask]
    boxes = boxes[mask]
    conf = conf[mask]
    cls_idxs = cls_idxs[mask]
    scores = scores[mask]

    # Filter by class if specified
    if classes is not None:
        class_mask = np.isin(cls_idxs, classes)
        if not np.any(class_mask):
            return (
                (np.zeros((0, ch + 1)), np.zeros((0,), dtype=int))
                if return_idxs
                else np.zeros((0, ch + 1))
            )
        x = x[class_mask]
        boxes = boxes[class_mask]
        conf = conf[class_mask]
        cls_idxs = cls_idxs[class_mask]
        scores = scores[class_mask]

    if x.shape[0] > max_nms:
        topk_idxs = np.argsort(-conf)[:max_nms]
        boxes = boxes[topk_idxs]
        x = x[topk_idxs]
        conf = conf[topk_idxs]
        cls_idxs = cls_idxs[topk_idxs]

    bboxes_cv2 = boxes.tolist()
    scores_cv2 = conf.tolist()

    indices = cv2.dnn.NMSBoxes(
        bboxes=bboxes_cv2,
        scores=scores_cv2,
        score_threshold=conf_thres,
        nms_threshold=iou_thres,
        top_k=max_det,
    )

    if len(indices) > 0:
        indices = np.array(indices).flatten()
        final = np.concatenate([xywh2xyxy(x[indices, :4]), x[indices, 4:]], axis=1)
        # Add predicted class index as the last column (like YOLO)
        final = np.concatenate(
            [final, cls_idxs[indices, None]], axis=1
        )  # shape: (N, ch+1)
    else:
        final = np.zeros((0, ch + 1))

    if time.time() - t_start > time_limit:
        print(f"NMS time limit {time_limit:.3f}s exceeded")

    return (final, indices) if return_idxs else final


def crop_mask_numpy(masks, boxes):
    """
    Crop masks to bounding boxes.

    Args:
        masks (np.ndarray): shape (n, h, w)
        boxes (np.ndarray): shape (n, 4), each box in relative coordinates (x1, y1, x2, y2) in pixels.

    Returns:
        np.ndarray: Cropped masks (same shape with values outside the box zeroed).
    """
    n, h, w = masks.shape
    cropped_masks = np.zeros_like(masks)

    for i in range(n):
        x1, y1, x2, y2 = boxes[i].astype(int)
        x1 = np.clip(x1, 0, w)
        x2 = np.clip(x2, 0, w)
        y1 = np.clip(y1, 0, h)
        y2 = np.clip(y2, 0, h)
        cropped_masks[i, y1:y2, x1:x2] = masks[i, y1:y2, x1:x2]

    return cropped_masks


def scale_masks_numpy(masks, shape, padding=True):
    """
    Resize masks to target shape (h, w).

    Args:
        masks (np.ndarray): (n, h, w)
        shape (tuple): target (height, width)
        padding (bool): whether to crop padded area before resizing

    Returns:
        np.ndarray: resized masks (n, shape[0], shape[1])
    """
    n, mh, mw = masks.shape
    new_h, new_w = shape

    gain = min(mh / new_h, mw / new_w)
    pad = [mw - new_w * gain, mh - new_h * gain]  # (pad_w, pad_h)

    if padding:
        pad_w, pad_h = pad[0] / 2, pad[1] / 2
        top, left = int(pad_h), int(pad_w)
        bottom, right = int(mh - pad_h), int(mw - pad_w)
        masks = masks[:, top:bottom, left:right]

    resized_masks = np.zeros((n, new_h, new_w), dtype=np.float32)
    for i in range(n):
        resized_masks[i] = cv2.resize(
            masks[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )

    return resized_masks


def clip_boxes_numpy(boxes, shape):
    """
    Clip bounding boxes to image shape.

    Args:
        boxes (np.ndarray): shape (..., 4)
        shape (tuple): (height, width)

    Returns:
        np.ndarray: clipped boxes
    """
    boxes = boxes.copy()
    boxes[..., 0] = np.clip(boxes[..., 0], 0, shape[1])  # x1
    boxes[..., 1] = np.clip(boxes[..., 1], 0, shape[0])  # y1
    boxes[..., 2] = np.clip(boxes[..., 2], 0, shape[1])  # x2
    boxes[..., 3] = np.clip(boxes[..., 3], 0, shape[0])  # y2
    return boxes


def scale_boxes_numpy(
    img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False
):
    """
    Scale boxes from one image shape to another.

    Args:
        img1_shape (tuple): (h1, w1)
        boxes (np.ndarray): shape (..., 4)
        img0_shape (tuple): (h0, w0)
        ratio_pad (tuple): ((gain,), (pad_x, pad_y))
        padding (bool): True if padded
        xywh (bool): boxes are in (x, y, w, h) format

    Returns:
        np.ndarray: scaled boxes
    """
    boxes = boxes.copy()

    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (
            (img1_shape[1] - img0_shape[1] * gain) / 2,
            (img1_shape[0] - img0_shape[0] * gain) / 2,
        )
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad[0]
        boxes[..., 1] -= pad[1]
        if not xywh:
            boxes[..., 2] -= pad[0]
            boxes[..., 3] -= pad[1]

    boxes[..., :4] /= gain
    return clip_boxes_numpy(boxes, img0_shape)


def process_mask_native_numpy(protos, masks_in, bboxes, shape):
    """
    Process mask using native numpy upsampling.

    Args:
        protos (np.ndarray): shape (c, h, w)
        masks_in (np.ndarray): shape (n, c)
        bboxes (np.ndarray): shape (n, 4)
        shape (tuple): (h, w)

    Returns:
        np.ndarray: shape (n, h, w)
    """
    c, mh, mw = protos.shape
    masks = np.dot(masks_in, protos.reshape(c, -1)).reshape(-1, mh, mw)
    masks = scale_masks_numpy(masks, shape)
    masks = crop_mask_numpy(masks, bboxes)
    return masks > 0


def process_mask_numpy(protos, masks_in, bboxes, shape, upsample=False):
    """
    Apply mask prototypes to bounding boxes.

    Args:
        protos (np.ndarray): shape (c, h, w)
        masks_in (np.ndarray): shape (n, c)
        bboxes (np.ndarray): shape (n, 4)
        shape (tuple): (h, w)
        upsample (bool): resize masks to original size

    Returns:
        np.ndarray: binary masks of shape (n, h, w)
    """
    c, mh, mw = protos.shape
    ih, iw = shape

    masks = np.dot(masks_in, protos.reshape(c, -1)).reshape(-1, mh, mw)

    width_ratio = mw / iw
    height_ratio = mh / ih

    scaled_bboxes = bboxes.copy()
    scaled_bboxes[:, [0, 2]] *= width_ratio
    scaled_bboxes[:, [1, 3]] *= height_ratio

    masks = crop_mask_numpy(masks, scaled_bboxes)

    if upsample:
        masks_resized = np.zeros((masks.shape[0], ih, iw), dtype=np.float32)
        for i in range(masks.shape[0]):
            masks_resized[i] = cv2.resize(
                masks[i], (iw, ih), interpolation=cv2.INTER_LINEAR
            )
        masks = masks_resized

    return masks > 0


def construct_results(preds, img, orig_imgs, protos):
    """
    Construct a list of result objects from the predictions.

    Args:
        preds (List[np.ndarray]): List of predicted bounding boxes, scores, and masks.
        img (np.ndarray): The image after preprocessing.
        orig_imgs (List[np.ndarray]): List of original images before preprocessing.
        protos (List[np.ndarray]): List of prototype masks.

    Returns:
        List[Tuple[np.ndarray, np.ndarray, np.ndarray]]: List of result tuples containing
        (original image, bounding boxes with scores and classes, masks)
    """
    return construct_result(preds, img, orig_imgs, protos)


def construct_result(pred, img, orig_img, proto):
    """
    Construct a single result object from the prediction.

    Args:
        pred (np.ndarray): The predicted bounding boxes, scores, and masks (N, 6+maskdim).
        img (np.ndarray): The image after preprocessing (C, H, W).
        orig_img (np.ndarray): The original image before preprocessing.
        proto (np.ndarray): The prototype masks.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A result tuple (original image, bounding boxes, masks).
    """
    if len(pred) == 0:
        masks = None
        return orig_img, np.zeros((0, 6), dtype=np.float32), masks
    if isinstance(pred, list):
        pred = pred[0] if len(pred) == 1 else np.concatenate(pred, axis=0)

    # if self.args.retina_masks:
    #     pred[:, :4] = scale_boxes_numpy(img.shape[1:], pred[:, :4], orig_img.shape[:2])  # from preprocessed to original
    #     masks = process_mask_native_numpy(proto, pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # (N, H, W)
    # else:
    masks = process_mask_numpy(
        proto, pred[:, 6:], pred[:, :4], img.shape[1:], upsample=True
    )  # (N, H, W)
    pred[:, :4] = scale_boxes_numpy(
        img.shape[1:], pred[:, :4], orig_img.shape[:2]
    )  # back to original scale

    if masks is not None:
        keep = masks.sum(axis=(1, 2)) > 0  # only keep predictions with non-zero masks
        pred, masks = pred[keep], masks[keep]

    return (orig_img, pred[:, :6], masks)


def post_process_seg(preds, img, orig_imgs, protos):
    preds = non_max_suppression_numpy(preds)
    results = construct_results(preds, img, orig_imgs, protos)
    return results


""" Các hàm để xử lý paddle input
"""


def preprocess_paddle_cls(image: np.ndarray, input_shape=[3, 48, 192]):
    """
    Enhanced preprocessing using OpenCV + NumPy only.

    Steps:
    1. Convert to RGB if needed
    2. Crop or pad width to target (384 px), keep original height
    3. Resize to model input shape (e.g. 48x192)
    4. Normalize using ImageNet stats

    Args:
        image: numpy array (HWC), BGR format (OpenCV)

    Returns:
        numpy array: shape (1, 3, H, W)
    """
    target_width = 384
    target_h, target_w = input_shape[1], input_shape[2]

    # Step 1: Ensure image is RGB
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    orig_h, orig_w = image.shape[:2]

    # Step 2: Crop or pad to target width
    if orig_w > target_width:
        # Center crop
        left = (orig_w - target_width) // 2
        image = image[:, left : left + target_width]
    elif orig_w < target_width:
        # Center pad
        pad_left = (target_width - orig_w) // 2
        pad_right = target_width - orig_w - pad_left
        image = cv2.copyMakeBorder(
            image, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

    # Step 3: Resize to target model input shape
    image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Step 4: Normalize
    image = image.astype(np.float32) / 255.0
    image = image.transpose(2, 0, 1)  # HWC to CHW
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    image = (image - mean) / std

    return np.expand_dims(image, axis=0).astype(np.float32)  # (1, 3, H, W)


def preprocess_paddle_det(img: np.ndarray, input_shape=[3, 640, 640]):
    """
    Preprocesses an image for PaddleOCR-style text detection models using OpenCV only.

    Args:
        img (np.ndarray): Input image in HWC format (RGB, BGR, or Grayscale).
        input_shape (list): Expected input shape of the model in CHW format, e.g., [3, 640, 640].

    Returns:
        tuple:
            - np.ndarray: Preprocessed image with shape (1, 3, H, W), dtype float32.
            - tuple: Original image height and width as (orig_h, orig_w).
            - float: Scale ratio applied during resizing.
    """
    # Handle image channels and convert to RGB
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    orig_h, orig_w = img.shape[:2]
    target_c, target_h, target_w = input_shape

    # Calculate scale ratio and new size
    scale_ratio = min(target_h / orig_h, target_w / orig_w)
    new_h, new_w = int(orig_h * scale_ratio), int(orig_w * scale_ratio)

    # Resize image
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad image to target size (top-left padding)
    padded_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    padded_img[:new_h, :new_w, :] = resized_img

    # Normalize image to float32 and apply mean-std normalization
    img_array = padded_img.astype("float32") / 255.0
    img_array -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
    img_array /= np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Change from HWC to CHW and add batch dimension
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, (orig_h, orig_w), scale_ratio


def postprocess_paddle_det(
    segmentation, original_shape, scale_ratio, pad_width=50, multi_heiht=3
):
    """
    Postprocess detection outputs to get expanded text boxes.

    The function doubles the height of each detected box and expands each side
    by a fixed 10 pixels. Boxes are scaled back to the original image size.

    Args:
        outputs (list): Model outputs.
        original_shape (tuple): (height, width) of original image.
        scale_ratio (float): Scaling ratio used during preprocessing.

    Returns:
        list: List of detected text boxes in format [[x1,y1,x2,y2,x3,y3,x4,y4], ...].
    """
    # pred = outputs[0]
    # segmentation = (pred >= 0.5).astype(np.uint8)[0, 0] * 255

    contours, _ = cv2.findContours(segmentation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes_with_sizes = []

    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # Scale box back to original image size
        box = box / scale_ratio

        # Calculate center, width and height
        center = np.mean(box, axis=0)
        edge1 = box[1] - box[0]
        edge2 = box[2] - box[1]

        width = np.linalg.norm(edge1)
        height = np.linalg.norm(edge2)

        if width < height:
            dummy = width
            width = height
            height = dummy

            dummyedge = edge1
            edge1 = edge2
            edge2 = dummyedge

        # Unit vectors for width and height direction
        width_dir = edge1 / width
        height_dir = edge2 / height

        # Expand width by 10px on both sides (20px total)
        new_width = width + pad_width
        # Double the height
        new_height = height * multi_heiht

        # Recalculate corner points
        new_box = np.array([
            center - width_dir * new_width / 2 - height_dir * new_height / 2,
            center + width_dir * new_width / 2 - height_dir * new_height / 2,
            center + width_dir * new_width / 2 + height_dir * new_height / 2,
            center - width_dir * new_width / 2 + height_dir * new_height / 2,
        ])

        # Clip to image boundaries
        new_box[:, 0] = np.clip(new_box[:, 0], 0, original_shape[1])
        new_box[:, 1] = np.clip(new_box[:, 1], 0, original_shape[0])

        # Optional filter for size
        boxes_with_sizes.append(new_box.reshape(-1).tolist())

    return boxes_with_sizes


def order_points_clockwise(pts):
    # Sort by x-coordinate
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    # Sort left-most by y to get top-left and bottom-left
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    tl, bl = left_most

    # Sort right-most by y to get top-right and bottom-right
    right_most = right_most[np.argsort(right_most[:, 1]), :]
    tr, br = right_most

    return np.array([tl, tr, br, bl], dtype=np.float32)


""" Các hàm để xử lý VietOCR
"""


class Vocab:
    def __init__(
        self,
        chars="aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ",
    ):
        self.pad = 0
        self.go = 1
        self.eos = 2
        self.mask_token = 3

        self.chars = chars

        self.c2i = {c: i + 4 for i, c in enumerate(chars)}

        self.i2c = {i + 4: c for i, c in enumerate(chars)}

        self.i2c[0] = "<pad>"
        self.i2c[1] = "<sos>"
        self.i2c[2] = "<eos>"
        self.i2c[3] = "*"

    def encode(self, chars):
        return [self.go] + [self.c2i[c] for c in chars] + [self.eos]

    def decode(self, ids):
        first = 1 if self.go in ids else 0
        last = ids.index(self.eos) if self.eos in ids else None
        sent = "".join([self.i2c[i] for i in ids[first:last]])
        return sent

    def __len__(self):
        return len(self.c2i) + 4

    def batch_decode(self, arr):
        texts = [self.decode(ids) for ids in arr]
        return texts

    def __str__(self):
        return self.chars


def resize(w, h, expected_height, image_min_width, image_max_width):
    new_w = int(expected_height * float(w) / float(h))
    round_to = 10
    new_w = math.ceil(new_w / round_to) * round_to
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)
    return new_w, expected_height


def process_image(
    image, image_height: int = 32, image_min_width: int = 32, image_max_width: int = 512
):
    """
    Args:
        image (np.ndarray): RGB image (H, W, 3)
        image_height (int): Target height
        image_min_width (int): Minimum width
        image_max_width (int): Maximum width

    Returns:
        np.ndarray: Normalized image of shape (3, H, W)
    """
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    new_w, image_height = resize(w, h, image_height, image_min_width, image_max_width)

    resized = cv2.resize(img, (new_w, image_height), interpolation=cv2.INTER_AREA)

    # Normalize
    resized = resized / 255.0

    # Convert to (C, H, W)
    resized = resized.transpose(2, 0, 1)

    return resized


def process_input(image, image_height, image_min_width, image_max_width):
    img = process_image(image, image_height, image_min_width, image_max_width)
    img = np.expand_dims(img, axis=0).astype(np.float32)  # shape: (1, C, H, W)
    return img


def translate_onnx(img, session, max_seq_length=128, sos_token=1, eos_token=2):
    """
    Translate input image using ONNX sessions (CNN + Transformer encoder-decoder).

    Args:
        img (np.ndarray): Input image of shape (B, C, H, W)
        session (tuple): Tuple of (cnn_session, encoder_session, decoder_session)
        max_seq_length (int): Maximum length for sequence decoding
        sos_token (int): Start-of-sequence token
        eos_token (int): End-of-sequence token

    Returns:
        np.ndarray: Translated sequences of shape (B, SeqLen)
    """
    cnn_session, encoder_session, decoder_session = session

    # Step 1: CNN feature extraction
    cnn_input = {cnn_session.get_inputs()[0].name: img}
    src = cnn_session.run(None, cnn_input)

    # Step 2: Transformer encoder
    encoder_input = {encoder_session.get_inputs()[0].name: src[0]}
    encoder_outputs, hidden = encoder_session.run(None, encoder_input)

    batch_size = img.shape[0]
    translated_sentence = [np.full(batch_size, sos_token, dtype=np.int64)]
    max_length = 0

    # Step 3: Autoregressive decoding loop
    while max_length <= max_seq_length and not np.all(
        np.any(np.stack(translated_sentence, axis=1) == eos_token, axis=1)
    ):
        tgt_inp = translated_sentence
        last_tokens = tgt_inp[-1]

        decoder_input = {
            decoder_session.get_inputs()[0].name: last_tokens,
            decoder_session.get_inputs()[1].name: hidden,
            decoder_session.get_inputs()[2].name: encoder_outputs,
        }

        output, hidden, _ = decoder_session.run(None, decoder_input)
        # output shape: (B, vocab_size)

        # Get top-1 token using numpy
        indices = np.argmax(output, axis=-1)
        translated_sentence.append(indices)
        max_length += 1

    translated_sentence = np.stack(translated_sentence, axis=1)
    return translated_sentence


""" PIPELINE
"""


def rotate_toward_angle(image: np.ndarray, angle: float):
    """
    Rotates an image (angle in degrees) and expands the image to avoid cropping.

    Args:
        image: Input image (as a NumPy array).
        angle: Angle in degrees. Positive values mean counter-clockwise rotation.

    Returns:
        Rotated image with adjusted size to contain the whole rotated content.
    """
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # Get rotation matrix
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)

    # Calculate the new bounding dimensions of the image
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to consider translation
    M[0, 2] += (new_w / 2) - cX
    M[1, 2] += (new_h / 2) - cY

    # Perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (new_w, new_h))


def is_vertical_box(boxes):
    w_sum, h_sum = 0, 0
    for box in boxes:
        box_np = np.array(box, dtype=np.int32).reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(box_np)
        w_sum += w
        h_sum += h
    if w_sum < h_sum or (w_sum / h_sum) < 2.5:
        return True
    else:
        return False


""" Log time """
# class Logger():
#     def __init__(self, logger_name="logger.txt"):
#         self.checkpoint={}
#         # with open(logger_name, mode="w", encoding="utf-8") as f:

#     def add_point(self):
#         current_time = time.time()
#         self.checkpoint.append(current_time)

#     def write_log(self):
#         pass


def letter_box(img: np.ndarray, target_size: tuple[int, int] | int, color=(0, 0, 0)):
    """
    Resize and pad image to target size while maintaining aspect ratio.

    Args:
        img (np.ndarray): Input image in HWC format.
        target_size (tuple or int): Target size as (width, height) or single int for square.
        color (tuple): Padding color.

    Returns:
        np.ndarray: Resized and padded image.
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    h, w = img.shape[:2]
    target_w, target_h = target_size

    if h * w > target_w * target_h:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized_img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    padded_img = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    padded_img[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized_img

    return padded_img, scale, (pad_w, pad_h)


def anyobb_to_tltrblbr(obb: np.ndarray):
    """
    Convert any order of 4 points of an oriented bounding box (OBB) to
    top-left, top-right, bottom-left, bottom-right order.

    Behavior
    --------
    - Find the center of the OBB
    - Determine the quadrant of each point relative to the center
    - Sort points based on their quadrant to achieve the desired order


    Parameters
    ----------
    - obb : `np.ndarray`
        - Array of shape (4, 2) representing the 4 corner points of the OBB.

    Returns
    -------
    - tltrblbr : `np.ndarray`
        - Array of shape (4, 2) representing the 4 corner points of the OBB in
        top-left, top-right, bottom-left, bottom-right order.
    """

    center = obb.mean(axis=0)
    pos_bool = obb > center
    pos_rank = pos_bool[:, 0] + pos_bool[:, 1] * 2
    pos_ordered = np.argsort(pos_rank)
    tltrblbr = obb[pos_ordered]
    return tltrblbr


def obb_letterbox_to_origin(obb_letter_box, padx, pady, scale):
    """
    Convert bounding box from letterbox coordinates back to original image coordinates.

    Behavior
    --------
    - Adjust for padding added during letterboxing
    - Scale coordinates back to original image size

    Parameters
    ----------
    - obb_letter_box : `np.ndarray`
        - Bounding box in letterbox coordinates
        - Format: (x1, y1, x2, y2, x3, y3, x4, y4)
        - Shape: (4, 2)
    - padx : `int`
        Padding in x direction.
    - pady : `int`
        Padding in y direction.
    - scale : `float`
        Scaling factor used during letterboxing.
        _description_

    Returns
    -------
    - obb_origin : `np.ndarray`
        - Bounding box in original image coordinates
        - Format: (x1, y1, x2, y2, x3, y3, x4, y4)
        - Shape: (4, 2)
    """
    obb_origin = obb_letter_box.copy()
    obb_origin[:, 0] -= padx
    obb_origin[:, 1] -= pady
    obb_origin /= scale
    return obb_origin


def get_best_obb(yolo_onnx_out):
    best_idx = yolo_onnx_out[:, 4].argsort()[::-1][0]
    best_box = yolo_onnx_out[best_idx].copy()
    if best_box[4] < 0.1:
        raise ValueError("No oriented bounding box detected.")
    best_box[-3:] = best_box[:-4:-1]
    best_box[[2, 3]] = best_box[[2, 3]]
    padding_result = OBB(best_box, orig_shape=(640, 640))
    det_obb = padding_result.xyxyxyxy[0]
    return det_obb


def obb_to_cropped(img: np.ndarray, bbox: np.ndarray, expand_ratio=1.2):
    """
    Crop and rotate image based on oriented bounding box (OBB).

    Parameters
    ----------
    - img : `np.ndarray`
        - Input image in HWC format.
    - bbox : `np.ndarray`
        - Oriented bounding box with shape (4,2) representing 4 corner points.
        - Format: (x1, y1, x2, y2, x3, y3, x4, y4)
    - expand_ratio : `float`
        - Ratio to expand the bounding box for cropping.

    Returns
    -------
    - cropped_img : `np.ndarray`
        - Cropped and rotated image region defined by the OBB.
    """
    tl, tr, bl, br = anyobb_to_tltrblbr(bbox)
    w = np.linalg.norm(bl - br).astype(np.int32).item()
    h = np.linalg.norm(bl - tl).astype(np.int32).item()
    origin_coord = np.float32([tl, tr, bl, br])
    new_coord = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    M = cv2.getPerspectiveTransform(origin_coord, new_coord)
    cropped = cv2.warpPerspective(img, M, (w, h))
    return cropped
