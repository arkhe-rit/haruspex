import cv2
import numpy as np


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left point has the smallest sum
    rect[2] = pts[np.argmax(s)]  # bottom-right point has the largest sum

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right point has the smallest difference
    rect[3] = pts[np.argmax(diff)]  # bottom-left point has the largest difference

    return rect

def perspective_transform(image, approx):
    approx = approx.reshape(4, 2)
    src_pts = order_points(approx)
    
    # Calculate the card's aspect ratio (assuming it's a vertical card)
    width_top = np.sqrt(((src_pts[1][0] - src_pts[0][0]) ** 2) + ((src_pts[1][1] - src_pts[0][1]) ** 2))
    width_bottom = np.sqrt(((src_pts[2][0] - src_pts[3][0]) ** 2) + ((src_pts[2][1] - src_pts[3][1]) ** 2))
    width = max(int(width_top), int(width_bottom))

    height_left = np.sqrt(((src_pts[3][0] - src_pts[0][0]) ** 2) + ((src_pts[3][1] - src_pts[0][1]) ** 2))
    height_right = np.sqrt(((src_pts[2][0] - src_pts[1][0]) ** 2) + ((src_pts[2][1] - src_pts[1][1]) ** 2))
    height = max(int(height_left), int(height_right))

    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype="float32")

    # Compute the homography matrix and apply the perspective transformation
    H, _ = cv2.findHomography(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, H, (width, height))

    return warped

def make_square(image):
    height, width = image.shape[:2]

    # Find the larger dimension (width or height)
    max_dim = max(width, height)

    # Resize the image to a square shape, stretching or squishing the content as needed
    square = cv2.resize(image, (max_dim, max_dim), interpolation=cv2.INTER_AREA)

    return square

def four_point_transform(image, pts):
    (tl, tr, br, bl) = order_points(pts)

    width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    max_width = max(int(width_top), int(width_bottom))

    height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    max_height = max(int(height_left), int(height_right))

    # Check the orientation of the card
    if max_width < max_height:
        max_width, max_height = max_height, max_width

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped