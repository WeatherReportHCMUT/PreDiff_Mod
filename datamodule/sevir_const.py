import numpy as np
import os

SEVIR_DATA_TYPES = ['vis', 'ir069', 'ir107', 'vil', 'lght']
SEVIR_RAW_DTYPES = {
    'vis': np.int16,
    'ir069': np.int16,
    'ir107': np.int16,
    'vil': np.uint8,
    'lght': np.int16
}
LIGHTING_FRAME_TIMES = np.arange(- 120.0, 125.0, 5) * 60
SEVIR_DATA_SHAPE = {
    'lght': (48, 48)
}

PREPROCESS_SCALE_SEVIR = {
    'vis': 1,  # Not utilized in original paper
    'ir069': 1 / 1174.68,
    'ir107': 1 / 2562.43,
    'vil': 1 / 47.54,
    'lght': 1 / 0.60517
}
PREPROCESS_OFFSET_SEVIR = {
    'vis': 0,  # Not utilized in original paper
    'ir069': 3683.58,
    'ir107': 1552.80,
    'vil': - 33.44,
    'lght': - 0.02990
}
PREPROCESS_SCALE_01 = {
    'vis': 1,
    'ir069': 1,
    'ir107': 1,
    'vil': 1 / 255,  # currently the only one implemented
    'lght': 1
}
PREPROCESS_OFFSET_01 = {
    'vis': 0,
    'ir069': 0,
    'ir107': 0,
    'vil': 0,  # currently the only one implemented
    'lght': 0
}

# TODO: UPdate this if the code move elsewhere
SEVIR_ROOT_DIR = '/root/sevir_data/sevir_full'
SEVIR_ROOT_LR_DIR = '/root/sevir_data/sevir_lr'
# sevir
SEVIR_CATALOG = os.path.join(SEVIR_ROOT_DIR, "CATALOG.csv")
SEVIR_DATA_DIR = os.path.join(SEVIR_ROOT_DIR, "data")
SEVIR_RAW_SEQ_LEN = 49
SEVIR_INTERVAL_REAL_TIME = 5
SEVIR_H_W_SIZE = (384,384)
# sevir-lr
SEVIR_LR_CATALOG = os.path.join(SEVIR_ROOT_LR_DIR, "CATALOG.csv")
SEVIR_LR_DATA_DIR = os.path.join(SEVIR_ROOT_LR_DIR, "data")
SEVIR_LR_RAW_SEQ_LEN = 25
SEVIR_LR_INTERVAL_REAL_TIME = 10
SEVIR_LR_H_W_SIZE = (128,128)

# region Constraints
SAMPLE_LIST = ['random', 'sequent']
LAYOUT_LIST = ('NHWT', 'NTHW', 'NTCHW', 'NTHWC', 'TNHW', 'TNCHW')
SPLIT_MODE_LIST = ('ceil', 'floor', 'uneven')
# endregion