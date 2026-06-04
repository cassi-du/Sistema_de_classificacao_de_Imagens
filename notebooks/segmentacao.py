import cv2
import numpy as np


def segmentar_grao(caminho):
    bgr = cv2.imread(caminho)
    if bgr is None:
        raise FileNotFoundError(f"Imagem não encontrada: {caminho}")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if num_labels <= 1:
        mask = closed
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_idx = np.argmax(areas) + 1
        mask = np.zeros_like(closed, dtype=np.uint8)
        mask[labels == max_idx] = 255

    return bgr, gray, mask

def desenhar_contorno(bgr, mask, color=(0, 255, 0), thickness=2):
    im = bgr.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return im
    c = max(contours, key=cv2.contourArea)
    cv2.drawContours(im, [c], -1, color, thickness)
    return im
