from pathlib import Path
import numpy as np
import cv2
import polanalyser as pa
import eventellipsometry as ee


def visualize_mueller_video(filename_npy):
    filename_npy = Path(filename_npy)
    dir_dst = filename_npy.parent
    filename_dst_stem = filename_npy.stem

    video_mm = np.load(filename_npy)

    # Crop ROI
    img_is_nan = np.isnan(video_mm).all(axis=(0, 3, 4))
    y0, y1 = np.where(~img_is_nan)[0][[0, -1]]
    x0, x1 = np.where(~img_is_nan)[1][[0, -1]]
    video_mm = video_mm[:, y0:y1, x0:x1]

    img_mueller_vis_list = [ee.mueller_image(pa.gammaCorrection(video_mm[i], 1 / 1), text_type="median", border=0, color_nan=(255, 255, 255)) for i in range(len(video_mm))]
    for i, img_mueller_vis in enumerate(img_mueller_vis_list):
        filename_png = f"{dir_dst}/images_median/{filename_dst_stem}_{i:03d}.png"
        Path(filename_png).parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(filename_png, img_mueller_vis)

    img_mueller_vis_list = [ee.mueller_image(pa.gammaCorrection(video_mm[i], 1 / 1), text_type="none", border=0, color_nan=(255, 255, 255)) for i in range(len(video_mm))]
    for i, img_mueller_vis in enumerate(img_mueller_vis_list):
        filename_png = f"{dir_dst}/images/{filename_dst_stem}_{i:03d}.png"
        Path(filename_png).parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(filename_png, img_mueller_vis)
