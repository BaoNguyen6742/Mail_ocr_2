import os

import cv2
import numpy as np
from paddleocr import DocImgOrientationClassification, PaddleOCR
from ultralytics import YOLO

from .utility import rotate_toward_angle


def get_config_path(filename):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "configs", filename)


def infer_det_yolo(yolo: YOLO, image: np.ndarray, device):
    results = yolo(image, device=device, imgsz=320)
    xyxy = None
    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            areas = []
            for box in boxes:
                xyxy = box.xyxy[0]
                area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                areas.append(area.item())

            largest_idx = areas.index(max(areas))
            largest_box = boxes[largest_idx]
            xyxy = largest_box.xyxy[0].cpu().numpy()

    x1, y1, x2, y2 = map(int, xyxy)
    cropped = image[y1:y2, x1:x2]
    return cropped


class OCR_server_pipeline:
    def __init__(
        self,
        yolo_seg_path,
        yolo_det_path,
        paddle_doc_cls_path,
        paddle_config,
        device="cuda",
    ):
        self.device = device
        (
            self.yolo_seg,
            self.yolo_det,
            self.paddle_doc_cls,
            self.ocr_model,
        ) = self.build_model(
            yolo_seg_path,
            yolo_det_path,
            paddle_doc_cls_path,
            paddle_config,
        )

    def build_model(
        self, yolo_seg_path, yolo_det_path, paddle_doc_cls_path, paddle_config
    ):
        yolo_seg = YOLO(yolo_seg_path, "segment")
        yolo_det = YOLO(
            yolo_det_path,
            "detect",
        )
        paddle_doc_cls = DocImgOrientationClassification(
            model_name="PP-LCNet_x1_0_doc_ori", model_dir=paddle_doc_cls_path
        )
        ocr_model = PaddleOCR(paddlex_config=paddle_config)

        return yolo_seg, yolo_det, paddle_doc_cls, ocr_model

    def ocr(self, image: np.ndarray):
        sentences = ""
        rotated_img = self.Quay_thu(image)
        cropped_img = self.Detec_address(rotated_img)
        sentences = self.ocr_model.predict(cropped_img)[0]["rec_texts"]
        return cropped_img, sentences

    def detect_text(self, image: np.ndarray):
        boxes = self.paddle_det.detect_text(image)
        return boxes

    def Quay_thu(self, image: np.ndarray):
        binary_mask = (
            self.yolo_seg(image, device=self.device, imgsz=320)[0]
            .masks.data[0]
            .cpu()
            .numpy()
            > 0.5
        ).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            raise ValueError("Không tìm thấy contour.")
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the angle of the rotated rectangle around the largest contour
        rect = cv2.minAreaRect(largest_contour)  # ((cx, cy), (w, h), angle)
        angle = rect[-1]

        # Adjust the angle to be within a certain range
        if angle < -45:
            angle += 90
        rotated_image = rotate_toward_angle(image, angle)
        rotated_mask = rotate_toward_angle(binary_mask, angle)
        rotated_mask = cv2.resize(
            rotated_mask,
            (rotated_image.shape[1], rotated_image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        x, y, w, h = cv2.boundingRect(rotated_mask)

        # Crop the rotated image and mask using the bounding box
        cropped_image = rotated_image[y : y + h, x : x + w]

        angle = int(self.paddle_doc_cls.predict(cropped_image)[0]["label_names"][0])

        final_image = rotate_toward_angle(cropped_image, angle)

        return final_image

    def Detec_address(self, image: np.ndarray):
        address = infer_det_yolo(self.yolo_det, image, self.device)
        return address
