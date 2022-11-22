import numpy as np
from .step1 import step1_basic_estimation
from .step2 import step2_final_estimation


def bm3d(img: np.ndarray, config: dict | None = None) -> np.ndarray:
    assert 'int' in str(img.dtype), 'Only integer images currently supported'

    if config is None:
        config = dict()
    sigma = config.get('sigma', 25.0)  # variance of the noise
    lamb2d = config.get('lamb2d', 2.0)
    lamb3d = config.get('lamb3d', 2.7)
    Step1_ThreDist = config.get('Step1_ThreDist', 2500)  # threshold distance
    Step1_MaxMatch = config.get('Step1_MaxMatch', 16)  # max matched blocks
    Step1_BlockSize = config.get('Step1_BlockSize', 8)
    Step1_WindowSize = config.get('Step1_WindowSize', 39)  # search window size
    Step2_ThreDist = config.get('Step2_ThreDist', 400)
    Step2_MaxMatch = config.get('Step2_MaxMatch', 32)
    Step2_BlockSize = config.get('Step2_BlockSize', 8)
    Step2_WindowSize = config.get('Step2_WindowSize', 39)
    Kaiser_Window_beta = config.get('Kaiser_Window_beta', 2.0)

    # ===============================================================================================

    # ============================================ BM3D =============================================
    param_step1 = tuple([Step1_BlockSize, Step1_ThreDist, Step1_MaxMatch, Step1_WindowSize,
                         lamb2d, lamb3d, sigma, Kaiser_Window_beta, 'cos'])

    param_step2 = tuple([Step2_BlockSize, Step2_ThreDist, Step2_MaxMatch, Step2_WindowSize,
                         sigma, Kaiser_Window_beta, 'cos'])

    # ==================================================================================================
    #                                         Basic estimate
    # ==================================================================================================
    basic_img = step1_basic_estimation(img, param_step1)
    # ==================================================================================================
    #                                         Final estimate
    # ==================================================================================================
    final_img = step2_final_estimation(basic_img, img, param_step2)
    return final_img
