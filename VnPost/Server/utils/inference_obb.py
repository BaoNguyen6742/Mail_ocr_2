from pathlib import Path

import cv2
import numpy as np
import torch
from paddleocr import DocImgOrientationClassification, PaddleOCR
from ultralytics import YOLO
from ultralytics.engine.results import OBB, Results

from .utility import anyobb_to_tltrblbr


def get_config_path(filename: str):
    """
    Get the absolute path to a configuration file.

    Behavior
    --------
    This function constructs the absolute path to a configuration file located in the "configs" directory relative to the current file's parent directory.

    Parameters
    ----------
    - filename : `str`
        The name of the configuration file.

    Returns
    -------
    -  : `str`
        The absolute path to the configuration file.
    """
    base_dir = Path(__file__).absolute().parent.parent
    base_dir.as_posix()
    return str((base_dir / "configs" / filename).as_posix())


def infer_obb_yolo(
    yolo_model,
    image: np.ndarray,
    device: str,
    pad_pixel: None | list[int] | int,
    pad_scale: None | list[float] | float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Infer the oriented bounding box (OBB) for the given image using the YOLO model.

    Behavior
    --------
    This function takes an image and passes it through the YOLO model to obtain the oriented bounding box.

    Parameters
    ----------
    - yolo_model : `YOLO`
        The YOLO model to use for inference.
    - image : `np.ndarray`
        The input image for which to infer the OBB. The image in format HWC with BGR color channel with pixel value unit8
    - device : `str`
        The device to run the inference on (e.g., "cuda" or "cpu").
    - pad_pixel : `None | list[int] | int`
        - The padding in pixel to add to the image before inference.
        - If a single value is provided, it will be used for both width and height padding.
    - pad_scale : `None | list[float] | float`
        - The padding in scale to add to the image before inference.
        - If a single value is provided, it will be used for both width and height padding.

    Returns
    -------
    - cropped : `np.ndarray`
        The cropped image containing the text area.
    - [tl, tr, br, bl] : `np.ndarray`
        The coordinates of the four corners of the text area.

    Raises
    ------
    ValueError
        If no oriented bounding box is detected.
    """
    if pad_pixel is not None and pad_scale is not None:
        raise ValueError("Only one of pad_pixel and pad_scale can be specified.")
    if (pad_pixel is not None) and (not isinstance(pad_pixel, list)):
        pad_pixel = [pad_pixel, pad_pixel]
    if (pad_scale is not None) and (not isinstance(pad_scale, list)):
        pad_scale = [pad_scale, pad_scale]

    result: Results = yolo_model(image, device=device, imgsz=320)[0]
    if result.obb is None:
        raise ValueError("No oriented bounding box detected.")
    obb: OBB = result.obb
    best_box = obb.conf.argmax().item()

    if pad_pixel is not None:
        obb_pad = torch.tensor([0, 0, *pad_pixel, 0, 0, 0])
        xywhr_conf_cls = result.obb.data.cpu()[best_box] + obb_pad
    if pad_scale is not None:
        obb_scale = torch.tensor([1, 1, *pad_scale, 1, 1, 1])
        xywhr_conf_cls = result.obb.data.cpu()[best_box] * obb_scale

    padding_result = OBB(xywhr_conf_cls, orig_shape=result.orig_shape)
    det_obb = padding_result.xyxyxyxy[best_box].cpu()
    tl, tr, bl, br = anyobb_to_tltrblbr(det_obb)

    w = torch.linalg.norm(bl - br).to(torch.int32).item()
    h = torch.linalg.norm(bl - tl).to(torch.int32).item()
    origin_coord = np.array([tl, tr, bl, br])
    new_coord = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    M = cv2.getPerspectiveTransform(origin_coord, new_coord)
    cropped = cv2.warpPerspective(image, M, (w, h))
    return cropped, [tl, tr, bl, br]


class OCR_server_pipeline:
    def __init__(
        self,
        yolo_obb_path,
        paddle_doc_cls_path,
        paddle_config,
        device="cuda",
    ):
        self.device = device
        (
            self.yolo_obb,
            self.paddle_doc_cls,
            self.ocr_model,
        ) = self.build_model(
            yolo_obb_path,
            paddle_doc_cls_path,
            paddle_config,
        )

    def build_model(
        self,
        yolo_obb_path,
        paddle_doc_cls_path,
        paddle_config,
    ):
        yolo_obb = YOLO(
            yolo_obb_path,
            "obb",
        )
        yolo_obb.compile()
        paddle_doc_cls = DocImgOrientationClassification(
            model_name="PP-LCNet_x1_0_doc_ori", model_dir=paddle_doc_cls_path
        )
        ocr_model = PaddleOCR(paddlex_config=paddle_config)

        return yolo_obb, paddle_doc_cls, ocr_model

    def ocr(
        self,
        image: np.ndarray,
        pad_pixel: None | int | list[int] = None,
        pad_scale: None | int | list[int] = None,
        debug=False,
    ):
        """
        Perform OCR on the given image.

        Behavior
        --------
        This function detects the address area in the image, classifies its orientation,
        and performs OCR on the cropped and rotated image.

        Parameters
        ----------
        - image : `np.ndarray`
            The input image to process.
        - pad_pixel : `None | int | List[int]`. Optional, by default None
            - The pixel padding to add to the image before inference.
            - If a single integer is provided, it will be used for both width and height padding.
        - pad_scale : `None | int | List[int]`. Optional, by default None
            - The scale padding to apply to the image before inference.
            - If a single float is provided, it will be used for both width and height padding.
        - debug : `bool`. Optional, by default False
            Whether to return debug information.

        Returns
        -------
        -  : `Tuple[np.ndarray, List[str]]`
            The cropped image and the recognized text.
        """

        addr_obb, tltrblbr = self._detec_addr_obb(
            image,
            pad_pixel,
            pad_scale,
        )
        angle = self._classify_angle(addr_obb)
        cropped_img = self._rotate_image(addr_obb, angle)
        sentences = self.ocr_model.predict(cropped_img)[0]["rec_texts"]
        if debug:
            return tltrblbr, angle, sentences
        return cropped_img, sentences

    def _detec_addr_obb(self, image: np.ndarray, pad_pixel, pad_scale):
        addr_obb = infer_obb_yolo(
            self.yolo_obb,
            image,
            self.device,
            pad_pixel,
            pad_scale,
        )
        return addr_obb

    def _classify_angle(self, image: np.ndarray):
        angle = int(self.paddle_doc_cls.predict(image)[0]["label_names"][0])
        return angle

    def _rotate_image(self, image: np.ndarray, angle: int):
        """
        Rotate the image based on the predicted angle.

        Behavior
        --------
        This function uses a document image orientation classification model to predict the angle of the input image. If the predicted angle is not zero, the image is rotated accordingly to correct its orientation.

        Parameters
        ----------
        - image : `np.ndarray`
            The input image to rotate.

        Returns
        -------
        - rotate_img : `np.ndarray`
            The rotated image.
        """
        if angle:
            rotate_img = cv2.rotate(image, int(3 - (angle / 90)))
        else:
            rotate_img = image
        return rotate_img
