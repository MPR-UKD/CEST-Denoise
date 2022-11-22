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
    KaiserWindowBeta = config.get('KaiserWindowBeta', 2.0)
    transformation_function = config.get('TransformationFunction', 'cos')

    assert transformation_function in ['cos', 'sin'], f'The transformation function {transformation_function} is ' \
                                                      f'not supported. Currently the following discrete ' \
                                                      f'transformation functions are supported: [cos, sin]'

    # Step 1 - basic estimate
    step1_threshold_distance = config.get('step1_threshold_distance', 2500)  # threshold distance
    step1_max_match = config.get('step1_max_match', 16)  # max matched blocks
    step1_BlockSize = config.get('step1_BlockSize', 8)  # BlockSize Step 1
    step1_WindowSize = config.get('step1_WindowSize', 39)  # search window size

    # Step 2 - final estimate
    step2_threshold_distance = config.get('step2_threshold_distance', 400)  # threshold distance
    step2_max_match = config.get('step2_max_match', 32)  # max matched blocks
    step2_BlockSize = config.get('step2_BlockSize', 8)  # BlockSize Step 2
    step2_WindowSize = config.get('step2_WindowSize', 39)  # search window size

    # ===============================================================================================

    # ============================================ BM3D =============================================
    param_step1 = tuple([step1_BlockSize, step1_threshold_distance, step1_max_match, step1_WindowSize,
                         lamb2d, lamb3d, sigma, KaiserWindowBeta, transformation_function])

    param_step2 = tuple([step2_BlockSize, step2_threshold_distance, step2_max_match, step2_WindowSize,
                         sigma, KaiserWindowBeta, transformation_function])

    # ==================================================================================================
    #                                         Basic estimate
    # ==================================================================================================
    basic_img = step1_basic_estimation(img, param_step1)

    # ==================================================================================================
    #                                         Final estimate
    # ==================================================================================================
    final_img = step2_final_estimation(basic_img, img, param_step2)

    return final_img
