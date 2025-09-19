import cv2
import numpy as np
import onnxruntime as ort

from .utility import (
    Vocab,
    anyobb_to_tltrblbr,
    get_best_obb,
    letter_box,
    obb_letterbox_to_origin,
    order_points_clockwise,
    postprocess_paddle_det,
    preprocess_paddle_det,
    process_input,
    translate_onnx,
)


class OCR_pipeline:
    def __init__(
        self,
        yolo_obb_onnx,
        paddle_cls_onnx,
        paddle_det_onnx,
        vietocr_cnn,
        vietocr_encoder,
        vietocr_decoder,
    ):
        self.yolo_obb_onnx = yolo_obb_onnx
        self.paddle_cls_onnx = paddle_cls_onnx
        self.paddle_det_onnx = paddle_det_onnx

        self.vietocr_cnn = vietocr_cnn
        self.vietocr_encoder = vietocr_encoder
        self.vietocr_decoder = vietocr_decoder

        self.vocab = Vocab()

    def ocr(self, image: np.ndarray):
        sentences = ""
        addr_img = self.Detec_addr_obb(image)
        angle = self.Classify_angle(addr_img)
        rotated = self._rotate_image(addr_img, angle)
        sentences = self.Regcognise_text(rotated)
        return rotated, sentences

    def Detec_addr_obb(self, image: np.ndarray):
        orig_h, orig_w = image.shape[:2]
        onnx_image, scale, padx, pady = self._preprocess_yolo_obb(image)
        yolo_onnx_out = self.yolo_obb_onnx.run(None, {"images": onnx_image})[
            0
        ].squeeze()
        addr_obb = self._post_process_yolo(image, padx, pady, scale, yolo_onnx_out)
        return addr_obb

    def Classify_angle(self, image: np.ndarray):
        image = self._preprocess_doc_cls(image)
        angle = self._infer_doc_cls_paddle(self.paddle_cls_onnx, image)
        return angle

    @staticmethod
    def _rotate_image(image: np.ndarray, angle):
        """
        Rotate the image based on the predicted angle.

        Behavior
        --------
        This function uses a document image orientation classification model to predict the angle of the input image. If the predicted angle is not zero, the image is rotated accordingly to correct its orientation.

        Parameters
        ----------
        - image : `np.ndarray`
            The input image to rotate.
        - angle : `int`
            The angle in clockwise direction predicted by the classification model. It should be one of {0, 90, 180, 270}.

        Returns
        -------
        - rotate_img : `np.ndarray`
            The rotated image.
        """
        height, width = image.shape[:2]
        if angle:
            rotate_img = cv2.rotate(image, int(3 - (angle / 90)))
        else:
            rotate_img = image
        return rotate_img

    def Regcognise_text(self, image: np.ndarray):
        sentences = ""

        boxes = self._infer_det_paddle(self.paddle_det_onnx, image)
        for i, box in enumerate(boxes):
            # Ensure box has shape (4, 2)
            box_np = np.array(box, dtype=np.float32).reshape(4, 2)

            # Sort the points to order: top-left, top-right, bottom-right, bottom-left
            rect = order_points_clockwise(box_np)

            # Compute width and height
            widthA = np.linalg.norm(rect[2] - rect[3])
            widthB = np.linalg.norm(rect[1] - rect[0])
            maxWidth = int(max(widthA, widthB))

            heightA = np.linalg.norm(rect[1] - rect[2])
            heightB = np.linalg.norm(rect[0] - rect[3])
            maxHeight = int(max(heightA, heightB))

            # Destination points in correct order
            dst = np.array(
                [
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1],
                ],
                dtype=np.float32,
            )

            # Perspective transform
            M = cv2.getPerspectiveTransform(rect, dst)
            cropped_region = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

            # cv2.imwrite(f"{i}.png", cropped_region)

            s = self._infer_rec_vietocr(
                self.vietocr_cnn,
                self.vietocr_encoder,
                self.vietocr_decoder,
                cropped_region,
            )
            s = self.vocab.decode(s)

            sentences = s + "\n" + sentences

        return sentences

    @staticmethod
    def _preprocess_yolo_obb(img: np.ndarray, image_size: int = 640):
        letter_boxed, scale, (pad_w, pad_h) = letter_box(img, target_size=image_size)
        cv2_img = cv2.cvtColor(letter_boxed, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
        onnx_img = np.ascontiguousarray(
            cv2_img.transpose(2, 0, 1).astype(np.float32)[None, ...] / 255
        )
        return onnx_img, scale, pad_w, pad_h

    @staticmethod
    def _post_process_yolo(origin_img, padx, pady, scale, yolo_onnx_out):
        det_obb = get_best_obb(yolo_onnx_out)
        origin_obb = obb_letterbox_to_origin(det_obb, padx, pady, scale)
        tl, tr, bl, br = anyobb_to_tltrblbr(origin_obb)
        w = np.linalg.norm(bl - br).astype(np.int32).item()
        h = np.linalg.norm(bl - tl).astype(np.int32).item()
        origin_coord = np.float32([tl, tr, bl, br])
        new_coord = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        M = cv2.getPerspectiveTransform(origin_coord, new_coord)
        cropped = cv2.warpPerspective(origin_img, M, (w, h))
        return cropped

    @staticmethod
    def _preprocess_doc_cls(img: np.ndarray, image_size=224):
        img_resized = cv2.resize(img, (224, 224))

        onnx_img = np.ascontiguousarray(
            img_resized.transpose(2, 0, 1).astype(np.float32)[None, ...] / 255.0
        )
        return onnx_img

    @staticmethod
    def _infer_doc_cls_paddle(session: ort.InferenceSession, image: np.ndarray):
        result = session.run(None, {"x": image})[0]
        angle = np.argmax(result) * 90
        return angle

    @staticmethod
    def _infer_det_paddle(
        session: ort.InferenceSession, image: np.ndarray, pad_width=50, mul_height=3
    ):
        """Perform inference using a Paddle-style ONNX text detection model.

        Args:
            session (ort.InferenceSession): ONNX runtime session for the detection model.
            image (np.ndarray): Input image in BGR format (uint8, shape: HxWx3).

        Returns:
            List[List[float]]: List of detected text boxes in 8-point format
                            [[x1, y1, x2, y2, x3, y3, x4, y4], ...]
        """
        # Step 1: Preprocess the input image for YOLO model
        img_input, original_shape, scale_ratio = preprocess_paddle_det(image)

        # Step 2: Run inference with the ONNX model
        outputs = session.run(None, {session.get_inputs()[0].name: img_input})[0]

        out1 = outputs.squeeze() >= 0.5
        segmentation = out1.astype(np.uint8) * 255

        boxes = postprocess_paddle_det(
            segmentation, original_shape, scale_ratio, pad_width, mul_height
        )

        return boxes

    @staticmethod
    def _infer_rec_vietocr(
        cnn: ort.InferenceSession,
        encoder: ort.InferenceSession,
        decoder: ort.InferenceSession,
        image: np.ndarray,
    ):
        """Perform inference using VietOCR

        Args:
            cnn (ort.InferenceSession): ONNX runtime session for the cnn model.
            encoder (ort.InferenceSession): ONNX runtime session for the encoder transformer model.
            decoder (ort.InferenceSession): ONNX runtime session for the decoder transformer model.
            image (np.ndarray): Input image in BGR format (uint8, shape: HxWx3).

        Returns:
            np.ndarray: Translated sequences of shape (B, SeqLen)
        """
        img = process_input(image, 32, 32, 512)

        session = (cnn, encoder, decoder)

        s = translate_onnx(np.array(img), session)[0].tolist()

        # vocab = Vocab()
        # s = vocab.decode(s)
        # print(s)
        return s
