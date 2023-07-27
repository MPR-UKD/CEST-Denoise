import numpy as np
from matplotlib import pyplot as plt
from .support_function import *
import itertools


def step2_final_estimation(
    basic_estimate_img: np.ndarray,  # 2D array of the basic estimate image
    noisy_image: np.ndarray,  # 2D array of the noisy image
    param: tuple,  # Tuple of hyperparameters for BM3D
    mask: np.ndarray,  # 2D array of the mask for the image
    verbose: bool = False,  # Flag for verbosity
) -> np.ndarray:
    """
    Give the final estimate after grouping, Wiener filtering and aggregation
    Return:
        final estimate finalImg
    """

    # Unpack the hyperparameters from the param tuple
    (
        BlockSize,
        ThresholdDistance,
        MaxMatch,
        WindowSize,
        sigma,
        KaiserWindowBeta,
        mode,
    ) = param

    # Initialize the final image, weights, and kaiser window arrays
    final_img, weights, kaiser_window = initialization(
        basic_estimate_img, BlockSize, KaiserWindowBeta
    )

    # Print the shapes of the final image and kaiser window arrays if verbose is True
    if verbose:
        print(f"Finale image shape: {final_img.shape}")
        print(f"Kaiser window shape: {kaiser_window.shape}")
        print(f"BlockSize: {BlockSize}")

    # Do discrete cosine / sinus transform for each block in the original image and the estimated image of step 1
    # to reduce the complexity
    if verbose:
        print(f"Run discrete {mode} 2D transformation")
    discrete_blocks_noisy = discrete_2D_transformation(noisy_image, BlockSize, mode)
    discrete_blocks_basic_estimate = discrete_2D_transformation(
        basic_estimate_img, BlockSize, mode
    )

    # block-wise calculation of final_image
    for x, y in itertools.product(
        range(int((basic_estimate_img.shape[0] - BlockSize))),
        range(int((basic_estimate_img.shape[1] - BlockSize))),
    ):
        ref_block_position = (x, y)
        if mask[x, y] == 0:
            continue

        # Grouping: Find similar blocks to current reference block in the estimated image.
        # Hint: Same block groups for basic_estimate_image and noisy_image
        block_positions, block_groups_basic_estimate, block_groups_noisy = grouping(
            basic_estimate_img,
            ref_block_position,
            BlockSize,
            ThresholdDistance,
            MaxMatch,
            WindowSize,
            discrete_blocks_basic_estimate,
            discrete_blocks_noisy,
        )

        block_groups_noisy, WienerWeight = filtering_3D(
            block_groups_basic_estimate, block_groups_noisy, sigma, mode
        )

        aggregation(
            block_groups_noisy,
            WienerWeight,
            block_positions,
            final_img,
            weights,
            kaiser_window,
        )

    weights = np.where(weights == 0, 1, weights)

    final_img[:, :] /= weights[:, :]

    if verbose:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(noisy_image, vmin=0, vmax=255)
        ax1.set_title("Noisy Image")
        ax2.imshow(basic_estimate_img, vmin=0, vmax=255)
        ax2.set_title("Basic Image")
        ax3.imshow(final_img, vmin=0, vmax=255)
        ax3.set_title("Final Image")
        plt.show()
    return final_img


def grouping(
    basic_estimate_image: np.ndarray,
    ref_block_position: tuple,
    block_size: int,
    threshold_distance: int,
    max_number_of_similar_blocks: int,
    window_size: int,
    discrete_transform_blocks_basic: np.ndarray,
    discrete_transform_blocks_noisy: np.ndarray,
):
    """
    find similar blocks in the estimated image based on the blocks after discrete transformation
    """

    # calculate search_window (np.array([[x1,y1],[x2,y2]])
    search_window = find_search_window(
        basic_estimate_image, ref_block_position, block_size, window_size
    )

    # number of blocks in site window
    num_blocks_in_window = (window_size - block_size + 1) ** 2

    # init block arrays and distance (necessary if more similar blocks find than max_number_of_similar_blocks)

    block_positions = np.zeros((num_blocks_in_window, 2), dtype=int)
    block_groups_basic = np.zeros(
        (num_blocks_in_window, block_size, block_size), dtype=float
    )
    block_groups_noisy = np.zeros(
        (num_blocks_in_window, block_size, block_size), dtype=float
    )
    distances = np.zeros(num_blocks_in_window, dtype=float)

    num_similar_blocks = 0

    # get current reference block from group of blocks
    reference_block = basic_estimate_image[
        ref_block_position[0] : ref_block_position[0] + block_size,
        ref_block_position[1] : ref_block_position[1] + block_size,
    ]

    for i in range(window_size - block_size + 1):
        for j in range(window_size - block_size + 1):
            current_block_position = (search_window[0, 0] + i, search_window[0, 1] + j)

            block = basic_estimate_image[
                current_block_position[0] : current_block_position[0] + block_size,
                current_block_position[1] : current_block_position[1] + block_size,
            ]

            dist = compute_distance_of_two_blocks(reference_block, block)

            # Add block and block position if block is similar (distance < threshold) to reference block
            if dist < threshold_distance:
                block_positions[num_similar_blocks, :] = current_block_position
                distances[num_similar_blocks] = dist
                num_similar_blocks += 1

    if num_similar_blocks <= max_number_of_similar_blocks:
        block_positions = block_positions[:num_similar_blocks, :]
    else:
        # more than max_number_of_similar_blocks similar blocks founded, return max_number_of_similar_blocks of
        # most similar blocks
        idx = np.argpartition(
            distances[:num_similar_blocks], max_number_of_similar_blocks
        )
        block_positions = block_positions[idx[:max_number_of_similar_blocks], :]

    for i in range(block_positions.shape[0]):
        similar_point = block_positions[i, :]
        block_groups_basic[i, :, :] = discrete_transform_blocks_basic[
            similar_point[0], similar_point[1], :, :
        ]
        block_groups_noisy[i, :, :] = discrete_transform_blocks_noisy[
            similar_point[0], similar_point[1], :, :
        ]

    block_groups_basic = block_groups_basic[: block_positions.shape[0], :, :]
    block_groups_noisy = block_groups_noisy[: block_positions.shape[0], :, :]

    return block_positions, block_groups_basic, block_groups_noisy


def filtering_3D(
    BlockGroup_basic: np.ndarray,
    block_groups_noisy: np.ndarray,
    sigma: float,
    mode: str,
):
    """
    Do collaborative Wiener filtering and here we choose 2D DCT + 1D DCT as the 3D transform which
    is the same with the 3D transform in hard-thresholding filtering
    """

    weight = 0

    for i in range(block_groups_noisy.shape[1]):
        for j in range(block_groups_noisy.shape[2]):
            # calculate wiener shrinkage coefficients
            # 1D-DCT \ DST again --> 3D-transform
            if mode == "cos":
                vec_basic = dct(BlockGroup_basic[:, i, j], norm="ortho")
                vec_noisy = dct(block_groups_noisy[:, i, j], norm="ortho")
            elif mode == "sin":
                vec_basic = dst(BlockGroup_basic[:, i, j], norm="ortho")
                vec_noisy = dst(block_groups_noisy[:, i, j], norm="ortho")
            else:
                raise NotImplementedError

            wiener_shrinkage_coef = vec_basic**2
            wiener_shrinkage_coef /= (
                wiener_shrinkage_coef + sigma**2
            )  # pixel weight -  wiener shrinkage coefficents

            vec_noisy *= wiener_shrinkage_coef

            #  sum up to get 2-norm squared of shrinkage coefficients (see eq. 11)
            weight += np.sum(wiener_shrinkage_coef**2)

            # 1D-back transformation
            if mode == "cos":
                block_groups_noisy[:, i, j] = list(idct(vec_noisy, norm="ortho"))
            elif mode == "sin":
                block_groups_noisy[:, i, j] = list(idst(vec_noisy, norm="ortho"))

    if weight > 0:
        wiener_weight = 1.0 / (sigma**2 * weight)
    else:
        wiener_weight = 1.0

    return block_groups_noisy, wiener_weight


def aggregation(
    block_groups_noisy,
    wiener_weight,
    block_positions,
    final_image,
    final_weight,
    finalKaiser,
):
    """
    Compute the final estimate of the true-image by aggregating all of the obtained local estimates
    using a weighted average
    """

    BlockWeight = wiener_weight * finalKaiser

    for i in range(block_positions.shape[0]):
        final_image[
            block_positions[i, 0] : block_positions[i, 0] + block_groups_noisy.shape[1],
            block_positions[i, 1] : block_positions[i, 1] + block_groups_noisy.shape[2],
        ] += BlockWeight * idct2D(block_groups_noisy[i, :, :])

        final_weight[
            block_positions[i, 0] : block_positions[i, 0] + block_groups_noisy.shape[1],
            block_positions[i, 1] : block_positions[i, 1] + block_groups_noisy.shape[2],
        ] += BlockWeight
