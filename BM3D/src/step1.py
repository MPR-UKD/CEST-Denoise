import itertools

import matplotlib.pyplot as plt

from .support_function import *
from typing import Tuple

# ==================================================================================================
#                                         Basic estimate
# ==================================================================================================


def step1_basic_estimation(
    noisy_img: np.ndarray, param: tuple, mask: np.ndarray, verbose: bool = False
):
    """
    Implementation of the first step of BM3D, which provides a base estimation of the image without noise
    after grouping, collaborative filtering and aggregation.

    Return:
        basic_estimate_img
    """
    # convert param to function variables
    assert len(param) == 9
    (
        BlockSize,
        ThresholdDistance,
        MaxMatch,
        WindowSize,
        lamb2d,
        lamb3d,
        sigma,
        KaiserWindowBeta,
        mode,
    ) = param

    # initialization of basic_estimate_img, weights (both np.zeros with same shape as noisy_image)
    # and the kaiser_window
    basic_estimate_img, weights, kaiser_window = initialization(
        noisy_img, BlockSize, KaiserWindowBeta
    )

    if verbose:
        print(f"Noisy image shape: {noisy_img.shape}")
        print(f"Kaiser window shape: {kaiser_window.shape}")
        print(f"BlockSize: {BlockSize}")

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
        range(int((noisy_img.shape[1] - BlockSize))),
    ):

        ref_block_position = (x, y)
        if mask[x, y] == 0:
            continue
        # Grouping: Find similar blocks to current reference block in the noisy \ orginal image, with Hard-Thresholding
        if verbose and x == 0 and y == 0:
            print(
                f"Search similar blocks with threshold distance = {ThresholdDistance}"
            )
        block_positions, block_groups = grouping(
            noisy_img,
            ref_block_position,
            discrete_blocks,
            BlockSize,
            ThresholdDistance,
            MaxMatch,
            WindowSize,
        )
        # Apply Hard 3D-Thresholding and filtering
        block_groups, nonzero_cnt = filtering_3d(block_groups, ThresholdValue3D, mode)

        # Aggregation: basic estimate of the true / denosiy image
        aggregation(
            block_groups,
            block_positions,
            basic_estimate_img,
            weights,
            kaiser_window,
            nonzero_cnt,
            sigma,
            mode,
        )

    weights = np.where(weights == 0, 1, weights)

    basic_estimate_img[:, :] /= weights[:, :]

    if verbose:
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(noisy_img, vmin=0, vmax=255)
        ax1.set_title("Noisy Image")
        ax2.imshow(basic_estimate_img, vmin=0, vmax=255)
        ax2.set_title("Basic Image")
        plt.show()

    return basic_estimate_img


def grouping(
    noisy_image: np.ndarray,
    ref_block_position: tuple,
    discrete_transform_blocks: np.ndarray,
    block_size: int,
    threshold_distance: int,
    max_number_of_similar_blocks: int,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    find similar block in the noisy image based on the blocks after discrete transformation
    """

    # calculate search_window (np.array([[x1,y1],[x2,y2]])
    search_window = find_search_window(
        noisy_image, ref_block_position, block_size, window_size
    )

    # number of blocks in site window
    num_blocks_in_window = (window_size - block_size + 1) ** 2

    # init block arrays and distance (necessary if more similar blocks find than max_number_of_similar_blocks)
    block_positions = np.zeros((num_blocks_in_window, 2), dtype=int)
    block_groups = np.zeros((num_blocks_in_window, block_size, block_size), dtype=float)
    distances = np.zeros(num_blocks_in_window, dtype=float)

    # get current reference block from group of blocks
    reference_block = discrete_transform_blocks[
        ref_block_position[0], ref_block_position[1], :, :
    ]

    num_similar_blocks = 0

    # Block searching and similarity (distance) computing
    for i in range(window_size - block_size + 1):
        for j in range(window_size - block_size + 1):
            # get temp block from group of blocks to calculate distance between the block and the reference block
            block = discrete_transform_blocks[
                search_window[0, 0] + i, search_window[0, 1] + j, :, :
            ]
            distance_block_to_reference = compute_distance_of_two_blocks(
                reference_block, block
            )

            # Add block and block position if block is similar (distance < threshold) to reference block
            if distance_block_to_reference < threshold_distance:
                block_positions[num_similar_blocks, :] = [
                    search_window[0, 0] + i,
                    search_window[0, 1] + j,
                ]
                block_groups[num_similar_blocks, :, :] = block
                distances[num_similar_blocks] = distance_block_to_reference
                num_similar_blocks += 1

    idx = np.argsort(distances)
    if num_similar_blocks <= max_number_of_similar_blocks:
        return (
            block_positions[idx[:num_similar_blocks], :],
            block_groups[idx[:num_similar_blocks], :, :],
        )
    else:
        # more than max_number_of_similar_blocks similar blocks founded, return max_number_of_similar_blocks of
        # most similar blocks
        return (
            block_positions[idx[:max_number_of_similar_blocks], :],
            block_groups[idx[:max_number_of_similar_blocks], :],
        )


def filtering_3d(
    block_group: np.ndarray, threshold: float, mode: str
) -> Tuple[np.ndarray, int]:
    """
    Perform collaborative hard-thresholding on the block group, which includes 3D transform,
    noise attenuation through hard-thresholding, and inverse 3D transform.
    """
    count = 0  # Counter for the number of non-zero elements in the transformed blocks

    # Loop over the x and y positions in the block group
    for x, y in itertools.product(
        range(block_group.shape[1]), range(block_group.shape[2])
    ):
        # Apply the discrete transform to the current block
        if mode == "cos":
            temp = dct(block_group[:, x, y], norm="ortho")
        elif mode == "sin":
            temp = dst(block_group[:, x, y], norm="ortho")
        else:
            # If the mode is not "cos" or "sin", raise an error
            raise NotImplementedError

        # Apply hard-thresholding to the transformed block
        temp[abs(temp[:]) < threshold] = 0.0

        # Update the non-zero element count
        count += np.nonzero(temp)[0].size
        # Apply the inverse transform to the thresholded block
        if mode == "cos":
            block_group[:, x, y] = list(idct(temp, norm="ortho"))
        elif mode == "sin":
            block_group[:, x, y] = list(idst(temp, norm="ortho"))
        else:
            # If the mode is not "cos" or "sin", raise an error
            raise NotImplementedError

    # Return the filtered block group and the non-zero element count
    return block_group, count


def aggregation(
    block_group: np.ndarray,  # 4D array of transformed blocks
    block_positions: np.ndarray,  # 2D array of block positions in the image
    basic_estimate_img: np.ndarray,  # 2D array to store the basic estimate image
    basic_weight: np.ndarray,  # 2D array to store the weights for each pixel in the basic estimate image
    basicKaiser: np.ndarray,  # 2D array of Kaiser window weights
    nonzero_cnt: int,  # Number of non-zero blocks in the block group
    sigma: float,  # Standard deviation of the noise
    mode: str,  # Mode for the discrete transform, either "cos" or "sin"
):
    # If there are no non-zero blocks, use the full block weight
    if nonzero_cnt < 1:
        block_weight = 1.0 * basicKaiser
    # Otherwise, scale the block weight by 1/(sigma^2 * nonzero_cnt)
    else:
        block_weight = (1.0 / (sigma**2 * nonzero_cnt)) * basicKaiser

    # Loop over the blocks in the block group
    for i in range(block_positions.shape[0]):
        # Apply the inverse discrete transform to the current block
        if mode == "cos":
            estimation = block_weight * idct2D(block_group[i, :, :])
        elif mode == "sin":
            estimation = block_weight * idst2D(block_group[i, :, :])
        else:
            # If the mode is not "cos" or "sin", raise an error
            raise NotImplementedError

        # Add the estimation to the basic estimate image and update the weights
        basic_estimate_img[
            block_positions[i, 0] : block_positions[i, 0] + block_group.shape[1],
            block_positions[i, 1] : block_positions[i, 1] + block_group.shape[2],
        ] += estimation

        basic_weight[
            block_positions[i, 0] : block_positions[i, 0] + block_group.shape[1],
            block_positions[i, 1] : block_positions[i, 1] + block_group.shape[2],
        ] += block_weight
