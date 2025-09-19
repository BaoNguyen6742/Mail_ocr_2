import os

import onnxruntime as ort

from .utility import *


def infer_det_yolo(
    session: ort.InferenceSession,
    image: np.ndarray,
    image_size: int = 640,
    padding: int = 10,
):
    """Run YOLO detection on an input image using an ONNX session and return the cropped region
    corresponding to the bounding box with the highest confidence score.

    Args:
        session (ort.InferenceSession): ONNX runtime inference session loaded with a YOLO model.
        image (np.ndarray): Original input image in BGR format.
        image_size (int): Size to resize the image for model input (e.g., 320, 640).

    Returns:
        np.ndarray | None: Cropped image region of the highest-confidence detection,
                           or None if no valid detection is found.
    """
    # Preprocess the image for YOLO model input
    img_input = preprocess_yolo_input(image, image_size)

    # Run inference
    outputs = session.run(None, {session.get_inputs()[0].name: img_input})
    detections = nms_classless(outputs[0])

    # Scale factors to map detection boxes back to original image size
    h0, w0 = image.shape[:2]
    scale_w, scale_h = w0 / image_size, h0 / image_size

    largest_conf = 0
    cropped = None

    # Iterate through detections and find the one with highest confidence
    for det in detections:
        x1, y1, x2, y2, conf = det
        if conf > largest_conf:
            x1 = int(x1 * scale_w)
            y1 = int(y1 * scale_h)
            x2 = int(x2 * scale_w)
            y2 = int(y2 * scale_h)

            # Apply padding and ensure coordinates are within image bounds
            x1_p = max(x1 - padding, 0)
            y1_p = max(y1 - padding, 0)
            x2_p = min(x2 + padding, image.shape[1])
            y2_p = min(y2 + padding, image.shape[0])

            # Crop the padded region from original image
            cropped = image[y1_p:y2_p, x1_p:x2_p]
            largest_conf = conf

    return cropped


def infer_seg_yolo(
    session: ort.InferenceSession, image: np.ndarray, image_size: int = 640
):
    """Perform segmentation inference using a YOLO model with ONNX runtime.

    Args:
    - session (ort.InferenceSession): ONNX runtime inference session.
    - image (np.ndarray): Input image for inference.
    - image_size (int): The size to which the input image should be resized for the model (default is 640).

    Returns:
    - np.ndarray: binary mask of segmentation.
    """
    # Step 1: Preprocess the input image for YOLO model
    img_input = preprocess_yolo_input(
        image, image_size
    )  # Resizes and normalizes the image

    # Step 2: Run inference with the ONNX model
    outputs = session.run(None, {session.get_inputs()[0].name: img_input})

    final_out2 = post_process_seg(outputs[0], img_input[0], image, outputs[1][0])
    # cv2.imwrite("testt.jpg",final_out2[2].transpose(1, 2, 0).astype(np.uint8)*255)

    binary_masks = final_out2[2].transpose(1, 2, 0).astype(np.uint8) * 255
    return binary_masks


def infer_cls_paddle(session: ort.InferenceSession, image: np.ndarray):
    """Runs angle classification inference using a Paddle-based ONNX model.

    Args:
        session (ort.InferenceSession): Initialized ONNX Runtime session for the classification model.
        image (np.ndarray): Input image as a NumPy array (HWC format, usually RGB or BGR).

    Returns:
        dict: A dictionary containing:
            - 'angle': int, the predicted rotation angle (0 or 180 degrees).
            - 'confidence': float, the confidence score of the predicted angle.
    """

    # Step 1: Preprocess the input image for YOLO model
    img_input = preprocess_paddle_cls(image)  # Resizes and normalizes the image

    # Step 2: Run inference with the ONNX model
    outputs = session.run(None, {session.get_inputs()[0].name: img_input})[0][0]

    if outputs[0] > outputs[1]:
        return {"angle": 0, "confidence": outputs[0]}
    else:
        return {"angle": 180, "confidence": outputs[1]}


def infer_det_paddle(
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
    outputs = session.run(None, {session.get_inputs()[0].name: img_input})

    out1 = outputs[0][0][0] >= 0.5
    segmentation = out1.astype(np.uint8) * 255

    boxes = postprocess_paddle_det(
        segmentation, original_shape, scale_ratio, pad_width, mul_height
    )

    return boxes


def infer_rec_vietocr(
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


class OCR_pipeline:
    def __init__(
        self,
        yolo_seg_onnx,
        yolo_det_onnx,
        paddle_cls_onnx,
        paddle_det_onnx,
        vietocr_cnn,
        vietocr_encoder,
        vietocr_decoder,
    ):
        self.yolo_seg_onnx = yolo_seg_onnx
        self.yolo_det_onnx = yolo_det_onnx
        self.paddle_cls_onnx = paddle_cls_onnx
        self.paddle_det_onnx = paddle_det_onnx

        self.vietocr_cnn = vietocr_cnn
        self.vietocr_encoder = vietocr_encoder
        self.vietocr_decoder = vietocr_decoder

        self.vocab = Vocab()

    def ocr(self, image: np.ndarray):
        sentences = ""
        rotated_img = self.Quay_thu(image)
        cropped_img = self.Detec_address(rotated_img)
        sentences = self.Nhan_dien(cropped_img)
        return cropped_img, sentences

    def Quay_thu(self, image: np.ndarray):
        binary_mask = infer_seg_yolo(self.yolo_seg_onnx, image)
        if binary_mask.ndim == 3 and binary_mask.shape[2] > 1:
            binary_mask = binary_mask[:, :, 0]  # Take the first channel
        binary_mask = binary_mask.astype(np.uint8)

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

        angle = 0
        boxes = infer_det_paddle(self.paddle_det_onnx, cropped_image)

        is_ver = is_vertical_box(boxes)

        if is_ver:
            angle = -90
        else:
            angle = 0

        vote_upsidedown = 0
        total_vote = 0
        # Crop each detected region
        for i, box in enumerate(boxes):
            # Convert box coordinates to numpy array
            box_np = np.array(box, dtype=np.int32).reshape(-1, 2)

            # Get the bounding rectangle of the rotated box
            x, y, w, h = cv2.boundingRect(box_np)

            # Crop the image (with some padding if needed)
            padding = 10  # Optional padding around text
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(cropped_image.shape[1], x + w + padding)
            y2 = min(cropped_image.shape[0], y + h + padding)

            cropped_region = cropped_image[y1:y2, x1:x2]
            if is_ver:
                cropped_region = cv2.rotate(cropped_region, cv2.ROTATE_90_CLOCKWISE)

            result = infer_cls_paddle(self.paddle_cls_onnx, cropped_region)

            if result["confidence"] > 0.89:
                total_vote += 1
                if result["angle"] != 0:
                    vote_upsidedown += 1

        posi = vote_upsidedown / total_vote

        if posi > 0.5:
            angle += 180

        final_img = rotate_toward_angle(cropped_image, angle)
        return final_img

    def Detec_address(self, image: np.ndarray):
        address = infer_det_yolo(self.yolo_det_onnx, image)
        return address

    def Nhan_dien(self, image: np.ndarray):
        sentences = ""

        boxes = infer_det_paddle(self.paddle_det_onnx, image)
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

            s = infer_rec_vietocr(
                self.vietocr_cnn,
                self.vietocr_encoder,
                self.vietocr_decoder,
                cropped_region,
            )
            s = self.vocab.decode(s)

            sentences = s + "\n" + sentences

        return sentences

    def ocr_custome(self, image: np.ndarray, img_name="", save_img_to="", f=None):
        sentences = ""
        # rotated_img = self.Quay_thu(image)
        # cropped_img = self.Detec_address(rotated_img)
        sentences = self.Nhan_dien_savedata(image, img_name, save_img_to, f)
        return image, sentences

    def Nhan_dien_savedata(
        self, image: np.ndarray, img_name="", save_img_to="", f=None
    ):
        folder_name = os.path.basename(save_img_to)

        sentences = ""
        boxes = infer_det_paddle(self.paddle_det_onnx, image, mul_height=4)
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

            # s = infer_rec_vietocr(self.vietocr_cnn, self.vietocr_encoder, self.vietocr_decoder, cropped_region)
            # s = self.vocab.decode(s)

            # save img and label file
            save_to = os.path.join(save_img_to, img_name + "_" + str(i) + ".png")
            cv2.imwrite(save_to, cropped_region)

            # if f is not None:
            #     f.write(f"{folder_name}/{img_name}_{str(i)}.jpg\t{s}\n")

            # sentences =  s + "\n" + sentences

        return sentences
