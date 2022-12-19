import itertools

import numpy as np
import matplotlib.pyplot as plt
from .support_function import *


# ==================================================================================================
#                                         Basic estimate
# ==================================================================================================


def step1_basic_estimation(noisy_img: np.ndarray,
                           param: tuple,
                           mask: np.ndarray,
                           verbose: bool = False):
    """
    Implementation of the first step of BM3D, which provides a base estimation of the image without noise
    after grouping, collaborative filtering and aggregation.

    Return:
        basic_estimate_img
    """
    # convert param to function variables
    assert len(param) == 9
    BlockSize, ThresholdDistance, MaxMatch, WindowSize, lamb2d, lamb3d, sigma, KaiserWindowBeta, mode = param

    # initialization of basic_estimate_img, weights (both np.zeros with same shape as noisy_image)
    # and the kaiser_window
    basic_estimate_img, weights, kaiser_window = initialization(noisy_img, BlockSize, KaiserWindowBeta)

    if verbose:
        print(f'Noisy image shape: {noisy_img.shape}')
        print(f'Kaiser window shape: {kaiser_window.shape}')
        print(f'BlockSize: {BlockSize}')

    # Do discrete cosine / sinus transform for each block in image to reduce the complexity
    # discrete_blocks.shape = (img.shape[0] - BlockSize, img.shape[1] - BlockSize, BlockSize, BlockSize)
    # Hint: discrete_blocks.shape = img.shape[0] - BlockSize, img.shape[1] - BlockSize, BlockSize, BlockSize
    if verbose:
        print(f"Run discrete {mode} 2D transformation")
    discrete_blocks = discrete_2D_transformation(noisy_img, BlockSize, mode)

    ThresholdValue3D = lamb3d * sigma

    # block-wise calculation of estimate_image
    for x, y in itertools.product(
            range(int((noisy_img.shape[0] - BlockSize))),
            range(int((noisy_img.shape[1] - BlockSize)))):

        ref_block_position = (x, y)
        if mask[x,y] == 0:
            continue
        # Grouping: Find similar blocks to current reference block in the noisy \ orginal image, with Hard-Thresholding
        if verbose and x == 0 and y == 0:
            print(f"Search similar blocks with threshold distance = {ThresholdDistance}")
        block_positions, block_groups = grouping(noisy_img, ref_block_position, discrete_blocks, BlockSize,
                                                 ThresholdDistance, MaxMatch, WindowSize)
        # Apply Hard 3D-Thresholding and filtering
        block_groups, nonzero_cnt = filtering_3d(block_groups, ThresholdValue3D, mode)

        # Aggregation: basic estimate of the true / denosiy image
        aggregation(block_groups, block_positions, basic_estimate_img, weights, kaiser_window, nonzero_cnt,
                    sigma, mode)

    weights = np.where(weights == 0, 1, weights)

    basic_estimate_img[:, :] /= weights[:, :]

    if verbose:
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(noisy_img, vmin=0, vmax=255)
        ax1.set_title('Noisy Image')
        ax2.imshow(basic_estimate_img, vmin=0, vmax=255)
        ax2.set_title('Basic Image')
        plt.show()

    return basic_estimate_img


def grouping(noisy_image: np.ndarray,
             ref_block_position: tuple,
             discrete_transform_blocks: np.ndarray,
             block_size: int,
             threshold_distance: int,
             max_number_of_similar_blocks: int,
             window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    find similar block in the noisy image based on the blocks after discrete transformation
    """

    # calculate search_window (np.array([[x1,y1],[x2,y2]])
    search_window = find_search_window(noisy_image, ref_block_position, block_size, window_size)

    # number of blocks in site window
    num_blocks_in_window = (window_size - block_size + 1) ** 2

    # init block arrays and distance (necessary if more similar blocks find than max_number_of_similar_blocks)
    block_positions = np.zeros((num_blocks_in_window, 2), dtype=int)
    block_groups = np.zeros((num_blocks_in_window, block_size, block_size), dtype=float)
    distances = np.zeros(num_blocks_in_window, dtype=float)

    # get current reference block from group of blocks
    reference_block = discrete_transform_blocks[ref_block_position[0], ref_block_position[1], :, :]

    num_similar_blocks = 0

    # Block searching and similarity (distance) computing
    for i in range(window_size - block_size + 1):
        for j in range(window_size - block_size + 1):
            # get temp block from group of blocks to calculate distance between the block and the reference block
            block = discrete_transform_blocks[search_window[0, 0] + i, search_window[0, 1] + j, :, :]
            distance_block_to_reference = compute_distance_of_two_blocks(reference_block, block)

            # Add block and block position if block is similar (distance < threshold) to reference block
            if distance_block_to_reference < threshold_distance:
                block_positions[num_similar_blocks, :] = [search_window[0, 0] + i, search_window[0, 1] + j]
                block_groups[num_similar_blocks, :, :] = block
                distances[num_similar_blocks] = distance_block_to_reference
                num_similar_blocks += 1

    if num_similar_blocks <= max_number_of_similar_blocks:
        return block_positions[:num_similar_blocks, :], block_groups[:num_similar_blocks, :, :]
    else:
        # more than max_number_of_similar_blocks similar blocks founded, return max_number_of_similar_blocks of
        # most similar blocks
        idx = np.argpartition(distances[:num_similar_blocks],
                              max_number_of_similar_blocks)
        return block_positions[idx[:max_number_of_similar_blocks], :], block_groups[idx[:max_number_of_similar_blocks], :]


def filtering_3d(block_group: np.ndarray,
                 threshold: float,
                 mode: str):
    # Do collaborative hard-thresholding which includes 3D transform, noise attenuation through
    # hard-thresholding and inverse 3D transform
    count = 0

    for x, y in itertools.product(range(block_group.shape[1]), range(block_group.shape[2])):
        if mode == 'cos':
            temp = dct(block_group[:, x, y], norm='ortho')
        elif mode == 'sin':
            temp = dst(block_group[:, x, y], norm='ortho')
        else:
            raise NotImplementedError

        temp[abs(temp[:]) < threshold] = 0.

        count += np.nonzero(temp)[0].size
        if mode == 'cos':
            block_group[:, x, y] = list(idct(temp, norm='ortho'))
        elif mode == 'sin':
            block_group[:, x, y] = list(idst(temp, norm='ortho'))
        else:
            raise NotImplementedError

    return block_group, count


def aggregation(block_group: np.ndarray,
                block_positions: np.ndarray,
                basic_estimate_img: np.ndarray,
                basic_weight: np.ndarray,
                basicKaiser: np.ndarray,
                nonzero_cnt: int,
                sigma: float,
                mode: str):
    if nonzero_cnt < 1:
        block_weight = 1.0 * basicKaiser
    else:
        block_weight = (1. / (sigma ** 2 * nonzero_cnt)) * basicKaiser

    for i in range(block_positions.shape[0]):
        if mode == 'cos':
            estimation = block_weight * idct2D(block_group[i, :, :])
        elif mode == 'sin':
            estimation = block_weight * idst2D(block_group[i, :, :])
        else:
            raise NotImplementedError
        basic_estimate_img[block_positions[i, 0]:block_positions[i, 0] + block_group.shape[1],
        block_positions[i, 1]:block_positions[i, 1] + block_group.shape[2]] += estimation

        basic_weight[block_positions[i, 0]:block_positions[i, 0] + block_group.shape[1],
        block_positions[i, 1]:block_positions[i, 1] + block_group.shape[2]] += block_weight
