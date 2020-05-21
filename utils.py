import base64
from typing import Union, Optional, List, Tuple

import cv2
import numpy as np
from matplotlib.colors import hsv_to_rgb
from scipy.ndimage import center_of_mass


def rle_encode(mask: np.ndarray) -> dict:
    """Perform Run-Length Encoding (RLE) on a binary mask.
    """

    assert mask.dtype == bool and mask.ndim == 2, 'RLE encoding requires a binary mask (dtype=bool).'
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return dict(data=base64.b64encode(runs.astype(np.uint32).tobytes()).decode('utf-8'), shape=mask.shape)


def rle_decode(rle: dict) -> np.ndarray:
    """Decode a Run-Length Encoding (RLE).
    """

    runs = np.frombuffer(base64.b64decode(rle['data']), np.uint32)
    shape = rle['shape']
    starts, lengths = [np.asarray(x, dtype=int) for x in (runs[0:][::2], runs[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def prm_visualize(
        instance_list: List[dict],
        class_names: Optional[List[str]] = None,
        font_scale: Union[int, float] = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Prediction visualization.
    """

    # helper functions
    def rgb2hsv(r, g, b):
        mx = max(r, g, b)
        mn = min(r, g, b)
        df = mx - mn
        if mx == mn:
            h = 0
        elif mx == r:
            h = (60 * ((g - b) / df) + 360) % 360
        elif mx == g:
            h = (60 * ((b - r) / df) + 120) % 360
        elif mx == b:
            h = (60 * ((r - g) / df) + 240) % 360
        s = 0 if mx == 0 else df / mx
        v = mx
        return h / 360.0, s, v

    def color_palette(N):
        cmap = np.zeros((N, 3))
        for i in range(0, N):
            uid = i
            r, g, b = 0, 0, 0
            for j in range(0, 8):
                r = np.bitwise_or(r, (((uid & (1 << 0)) != 0) << 7 - j))
                g = np.bitwise_or(g, (((uid & (1 << 1)) != 0) << 7 - j))
                b = np.bitwise_or(b, (((uid & (1 << 2)) != 0) << 7 - j))
                uid = (uid >> 3)
            cmap[i, 0] = min(r + 86, 255)
            cmap[i, 1] = min(g + 86, 255)
            cmap[i, 2] = b
        cmap = cmap.astype(np.float32) / 255
        return cmap

    if len(instance_list) > 0:
        palette = color_palette(len(instance_list) + 1)
        height, width = instance_list[0]['mask'].shape[0], instance_list[0]['mask'].shape[1]
        instance_mask = np.zeros((height, width, 3), dtype=np.float32)
        peak_response_map = np.zeros((height, width, 3), dtype=np.float32)
        for idx, pred in enumerate(instance_list):
            category, mask, prm = pred['category'], pred['mask'], pred['iam']
            # instance masks
            instance_mask[mask, 0] = palette[idx + 1][0]
            instance_mask[mask, 1] = palette[idx + 1][1]
            instance_mask[mask, 2] = palette[idx + 1][2]
            if class_names is not None:
                y, x = center_of_mass(mask)
                y, x = int(y), int(x)
                text = class_names[category]
                font_face = cv2.FONT_HERSHEY_SIMPLEX
                thickness = 2
                text_size, _ = cv2.getTextSize(text, font_face, font_scale, thickness)
                cv2.putText(
                    instance_mask,
                    text,
                    (x - text_size[0] // 2, y),
                    font_face,
                    font_scale,
                    (1., 1., 1.),
                    thickness)
            # peak response map
            peak_response = (prm - prm.min()) / (prm.max() - prm.min())
            mask = peak_response > 0.01
            h, s, _ = rgb2hsv(palette[idx + 1][0], palette[idx + 1][1], palette[idx + 1][2])
            peak_response_map[mask, 0] = h
            peak_response_map[mask, 1] = s
            peak_response_map[mask, 2] = np.power(peak_response[mask], 0.5)

        peak_response_map = hsv_to_rgb(peak_response_map)
        return instance_mask, peak_response_map
