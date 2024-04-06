import numpy as np
IMAGE_MEANS =np.array([117.67, 130.39, 121.52, 162.92]) # The setting here is for Chesapeake dataset
IMAGE_STDS = np.array([39.25,37.82,24.24,60.03])
LABEL_CLASSES = [0, 11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95] 
LABEL_CLASS_COLORMAP = { # Color map for Chesapeake dataset
    0:  (0, 0, 0),
    11: (70, 107, 159),
    12: (209, 222, 248),
    21: (222, 197, 197),
    22: (217, 146, 130),
    23: (235, 0, 0),
    24: (171, 0, 0),
    31: (179, 172, 159),
    41: (104, 171, 95),
    42: (28, 95, 44),
    43: (181, 197, 143),
    52: (204, 184, 121),
    71: (223, 223, 194),
    81: (220, 217, 57),
    82: (171, 108, 40),
    90: (184, 217, 235),
    95: (108, 159, 184)
}

LABEL_IDX_COLORMAP = {
    idx: LABEL_CLASS_COLORMAP[c]
    for idx, c in enumerate(LABEL_CLASSES)
}

def get_label_class_to_idx_map():
    label_to_idx_map = []
    idx = 0
    for i in range(LABEL_CLASSES[-1]+1):
        if i in LABEL_CLASSES:
            label_to_idx_map.append(idx)
            idx += 1
        else:
            label_to_idx_map.append(0)
    label_to_idx_map = np.array(label_to_idx_map).astype(np.int64)
    return label_to_idx_map

LABEL_CLASS_TO_IDX_MAP = get_label_class_to_idx_map()