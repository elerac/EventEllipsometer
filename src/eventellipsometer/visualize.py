import argparse
from pathlib import Path
import numpy as np
import cv2
import polanalyser as pa
from .utils_mueller import mueller_image


def save_images(video_mm, dir_dst, filename_dst_stem, subfolder, text_type):
    img_mueller_vis_list = [mueller_image(pa.gammaCorrection(frame, 1 / 1), text_type=text_type, border=0, color_nan=(255, 255, 255)) for frame in video_mm]
    for i, img_mueller_vis in enumerate(img_mueller_vis_list):
        filename_png = f"{dir_dst}/{subfolder}/{filename_dst_stem}_{i:03d}.png"
        Path(filename_png).parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(filename_png, img_mueller_vis)


def crop_roi(video_mm, axis=(0, 3, 4)):
    img_is_nan = np.isnan(video_mm).all(axis=axis)
    y0, y1 = np.where(~img_is_nan)[0][[0, -1]]
    x0, x1 = np.where(~img_is_nan)[1][[0, -1]]
    video_mm = video_mm[:, y0:y1, x0:x1]
    return video_mm


def visualize_mueller_video(filename_npy):
    filename_npy = Path(filename_npy)
    dir_dst = filename_npy.parent
    filename_dst_stem = filename_npy.stem

    video_mm = np.load(filename_npy)

    # Without crop ROI
    save_images(video_mm, dir_dst, filename_dst_stem, "images_median_wo_ROI", "median")
    save_images(video_mm, dir_dst, filename_dst_stem, "images_wo_ROI", "none")

    # Crop ROI
    video_mm = crop_roi(video_mm)

    save_images(video_mm, dir_dst, filename_dst_stem, "images_median", "median")
    save_images(video_mm, dir_dst, filename_dst_stem, "images", "none")
    video_mm_05 = video_mm.copy()
    video_mm_05[..., 0, 0] *= 0.5
    save_images(video_mm_05, dir_dst, filename_dst_stem, "images_05", "none")

    video_mm_025 = video_mm.copy()
    video_mm_025[..., 0, 0] *= 0.25
    save_images(video_mm_025, dir_dst, filename_dst_stem, "images_025", "none")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename_npy", type=str)
    args = parser.parse_args()
    visualize_mueller_video(args.filename_npy)


if __name__ == "__main__":
    main()
