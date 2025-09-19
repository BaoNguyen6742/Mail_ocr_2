import cv2
import numpy as np


def rotate_toward_angle(image: np.ndarray, angle: float):
    """
    Rotates an image (angle in degrees) and expands the image to avoid cropping.

    Args:
        image: Input image (as a NumPy array).
        angle: Angle in degrees. Positive values mean counter-clockwise rotation.

    Returns:`
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
