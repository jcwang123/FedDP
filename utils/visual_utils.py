import cv2
import numpy as np


def draw_contour(image: np.array, pred: np.array, color):
    assert len(image.shape) == 3
    assert len(pred.shape) == 2

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, -1)

    contours, hierarchy = cv2.findContours(pred, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    image = image.copy()
    image = cv2.drawContours(image, contours, -1, color, 1)
    print(image.shape)
    return image


def draw_mask(image: np.array, label: np.array, rate: float):
    kernel = np.ones((5, 5), np.uint8)
    label = cv2.erode(label.astype('uint8'), kernel, iterations=1)
    label = label * 255
    label = np.repeat(label[..., np.newaxis], 3, -1)
    tmp = image * rate + label * (1 - rate)
    tmp = tmp.astype('uint8')
    return tmp


# import seaborn as sns
# import matplotlib as mpl

# pal = sns.color_palette(palette='bright')
# print(pal)