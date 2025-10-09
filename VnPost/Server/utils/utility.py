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


def anyobb_to_tltrblbr(
    obb: np.ndarray[tuple[int, int], np.dtype[np.float32]],
) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
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
    - obb : `np.ndarray[tuple[int, int], np.dtype[np.float32]]`
        - Array representing the 4 corner points of the OBB.
        - Shape: (4, 2), where each row is a point (x, y).
        - Dtype: float32

    Returns
    -------
    - tltrblbr : `np.ndarray[tuple[int, int], np.dtype[np.float32]]`
        - Array representing the 4 corner points of the OBB in
        top-left, top-right, bottom-left, bottom-right order.
        - Shape: (4, 2), where each row is a point (x, y).
        - Dtype: float32
    """

    center = obb.mean(axis=0)
    pos_bool = obb > center
    pos_rank = pos_bool[:, 0] + pos_bool[:, 1] * 2
    pos_ordered = np.argsort(pos_rank)
    tltrblbr = obb[pos_ordered]
    return tltrblbr
