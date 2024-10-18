import polanalyser as pa
import numpy as np
import cv2
from typing import Sequence


def mueller_image(M: np.ndarray, colormap: str = "RdBu", text_type: str = "median", block_size: int = 64, border: int = 2, color_nan: Sequence[int] = (128, 128, 128)) -> np.ndarray:
    """Visualize the Mueller matrix as an image.

    Parameters
    ----------
    M : np.ndarray
        The Mueller matrix. The shape should be (3, 3) or (4, 4) or (H, W, 3, 3) or (H, W, 4, 4).
    colormap : str, optional
        The color map for visualization. The default is "RdBu".
    text_type : bool, optional
        The type of text. {"median", "mean", "center", "none"}. "none" means no text. The default is "median".
    block_size : int, optional
        The size of the block. This is used when the input matrix is 1x1. The default is 64.
    border : int, optional
        The border size. The default is 2.
    color_nan : Sequence[int], optional
        The color for NaN values. The default is (128, 128, 128).

    Returns
    -------
    np.ndarray
        The image of the Mueller matrix.
    """

    # Check the input matrix
    if M.ndim == 2:
        if M.shape not in [(3, 3), (4, 4)]:
            raise ValueError(f"The Mueller matrix should be (3, 3) or (4, 4), but got {M.shape}")
    elif M.ndim == 4:
        if M.shape[2:] not in [(3, 3), (4, 4)]:
            raise ValueError(f"The Mueller matrix should be (3, 3) or (4, 4), but got {M.shape}")
    else:
        raise ValueError(f"The Mueller matrix should be (3, 3), (4, 4), (H, W, 3, 3), (H, W, 4, 4), but got {M.shape}")

    # Repeat for blocks
    if M.ndim == 2:
        M = M[None, None, ...]  # (1, 1, 3, 3) or (1, 1, 4, 4)
        M = np.repeat(M, block_size, axis=0)
        M = np.repeat(M, block_size, axis=1)  # (H, W, 3, 3) or (H, W, 4, 4)

    # Normalize the matrix
    M = M / M[:, :, 0, 0][:, :, None, None]

    # Apply color map
    M_vis = pa.applyColorMap(M, colormap, -1, 1)

    M_vis[np.isnan(M)] = color_nan  # Set invalid values to gray

    # Write number in text for each block at the center
    if text_type != "none":
        H, W, MH, MW = M.shape

        if text_type == "median":
            M_vals = np.nanmedian(M, axis=(0, 1))
        elif text_type == "mean":
            M_vals = np.nanmean(M, axis=(0, 1))
        elif text_type == "center":
            M_vals = M[H // 2, W // 2]
        else:
            raise ValueError(f"Invalid text_type: {text_type}")

        for j in range(MH):
            for i in range(MW):
                img_M_vis_block = M_vis[:, :, j, i].copy()
                text = f"{M_vals[j, i]:.2f}"
                text = text.replace("-0.00", "0.00")
                font = cv2.FONT_HERSHEY_COMPLEX
                thickness = 1
                fontScale = cv2.getFontScaleFromHeight(font, int(min(H, W) * 0.2), thickness)
                textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]

                # get coords based on boundary
                textX = (W - textsize[0]) // 2
                textY = (H + textsize[1]) // 2
                # add text centered on image
                img_M_vis_block = cv2.putText(
                    img_M_vis_block,
                    text,
                    (textX, textY),
                    font,
                    fontScale,
                    (0, 0, 0),
                    thickness,
                    cv2.LINE_AA,
                )
                M_vis[:, :, j, i] = img_M_vis_block

    # Make grid
    img_M_vis_grid = pa.makeGridMueller(M_vis, border=border)

    return img_M_vis_grid


def imwrite_mueller(filename: str, M: np.ndarray, text_type: str = "median"):
    img = mueller_image(M, text_type=text_type)
    cv2.imwrite(filename, img)
