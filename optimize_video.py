import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import polanalyser as pa
import eventellipsometry as ee
from recordings.filenames import filename_raw


def main():
    np.set_printoptions(precision=3, suppress=True)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--filename_raw", type=str, default=filename_raw)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--max_iter_svd", type=int, default=10)
    parser.add_argument("--tol", type=float, default=0.01)
    parser.add_argument("--max_iter_propagate", type=int, default=10)
    args = parser.parse_args()

    print("Read event data")
    print(f"  {args.filename_raw}")
    events = ee.read_event(args.filename_raw)

    bias_diff = int(events["bias_diff"])
    bias_diff_on = int(events["bias_diff_on"])
    bias_diff_off = int(events["bias_diff_off"])
    print(f"  Bias diff: {bias_diff}")
    print(f"  Bias diff on: {bias_diff_on}")
    print(f"  Bias diff off: {bias_diff_off}")
    width = events["width"]
    height = events["height"]

    # Triggers (jus for checking)
    trig_t = events["trig_t"]
    trig_p = events["trig_p"]
    trig_t_x1 = trig_t[trig_p == 0]  # OFF
    trig_t_x5 = trig_t[trig_p == 1]  # ON
    trig_t_x1, trig_t_x5 = ee.clean_triggers(trig_t_x1, trig_t_x5)
    period = np.diff(trig_t_x1)
    period = np.append(period, period[-1])  # Add last element
    trig_t_diff = trig_t_x5 - trig_t_x1  # [0, period / 5]
    phi_offsets = trig_t_diff / period * np.pi
    print(f"  Triggers: {len(trig_t_x1)=}, {len(trig_t_x5)=}")
    print(f"  Avg. frequency: {1 / np.mean(period) * 1000000:.2f} fps")
    # for i, (trig_t_x1_i, trig_t_x5_i, trig_t_diff_i, period_i, phi_offset_i) in enumerate(zip(trig_t_x1, trig_t_x5, trig_t_diff, period, phi_offsets)):
    #     print(f"Triggers {i}: {trig_t_x1_i:d}, {trig_t_x5_i:d}, {trig_t_diff_i:d}, {period_i:d}, {np.rad2deg(phi_offset_i):.2f}")

    # Calib
    phi1 = np.deg2rad(56)
    phi2 = np.deg2rad(146)

    # phi1 = np.deg2rad(6.5)
    # phi2 = np.deg2rad(64)

    phi1 = np.deg2rad(6.5)
    phi2 = np.deg2rad(62.5)

    # filename_calib_phi1 = "calib/qwp_offset/phi1.txt"
    # filename_calib_phi2 = "calib/qwp_offset/phi2.txt"
    # phi1 = float(np.loadtxt(filename_calib_phi1, comments="%"))
    # phi2 = float(np.loadtxt(filename_calib_phi2, comments="%"))

    print("Calibration parameters")
    print(f"  QWP offset")
    print(f"    phi1: {np.rad2deg(phi1):.2f} ({phi1:.2f})")
    print(f"    phi2: {np.rad2deg(phi2):.2f} ({phi2:.2f})")
    filename_C_on = f"calib/thresh/on/diff{bias_diff}_diff_on{bias_diff_on}_diff_off{bias_diff_off}.npy"
    filename_C_off = f"calib/thresh/off/diff{bias_diff}_diff_on{bias_diff_on}_diff_off{bias_diff_off}.npy"
    print(f"  Thresholds")
    print(f"    Con : {filename_C_on}")
    print(f"    Coff: {filename_C_off}")
    img_C_on = np.load(filename_C_on).astype(np.float32)
    img_C_off = np.load(filename_C_off).astype(np.float32)

    print("Construct dataframes")
    ellipsometry_eventmaps = ee.construct_dataframes(events["x"], events["y"], events["t"], events["p"], width, height, trig_t, trig_p, img_C_on, img_C_off)
    if args.max_frames is not None:
        ellipsometry_eventmaps = ellipsometry_eventmaps[: args.max_frames]
    print(f"  Number of frames: {len(ellipsometry_eventmaps)}")

    print("Start reconstruction")
    time_start = time.time()
    video_mm = ee.fit_mueller(
        ellipsometry_eventmaps,
        phi1,
        phi2,
        max_iter_svd=args.max_iter_svd,
        tol=args.tol,
        max_iter_propagate=args.max_iter_propagate,
        verbose=True,
    )
    video_mm = np.array(video_mm)
    print(f"Total reconstruction time: {time.time() - time_start:.2f} s")

    print("Save Mueller matrix video")
    filename_npz = Path("mmvideo") / f"{Path(args.filename_raw).stem}_iter_svd{args.max_iter_svd}_tol{args.tol}_iter_propagate{args.max_iter_propagate}.npz"
    filename_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(filename_npz.with_suffix(".npz"), video_mm=video_mm)
    print(f"  {filename_npz}")
    exit()

    # Visualize
    num_frames, height, width, _, _ = video_mm.shape
    video_mae = np.full((num_frames, height, width), np.nan)  # Video of mean absolute error
    video_color = np.zeros((num_frames, height, width, 3))  # Video of color
    for f in range(num_frames):
        for y in range(height):
            for x in range(width):
                theta, dlogI, weights, phi_offset = ellipsometry_eventmaps[f].get(x, y)
                if len(theta) == 0:
                    continue

                n_coffs, d_coffs = ee.calculate_ndcoffs(theta, phi1, phi2 - 5 * phi_offset)
                M = video_mm[f, y, x]
                dlogI_pred = (n_coffs @ M.flatten()) / (d_coffs @ M.flatten())
                r = np.abs(dlogI - dlogI_pred)
                mae = np.mean(r * weights)
                video_mae[f, y, x] = mae

                p = np.sign(dlogI)
                color_pos = np.array([0.0, 0.0, 1.0])
                color_neg = np.array([1.0, 0.0, 0.0])
                # inv_r = 1 / r
                # inv_r_normalized = inv_r
                # r_pecentile = np.percentile(r, 99)
                # invalid = r > r_pecentile
                w_pos = (p == 1) / p.size * r**2 / 100
                w_neg = (p == -1) / p.size * r**2 / 100
                color = color_pos * w_pos[:, None] + color_neg * w_neg[:, None]
                video_color[f, y, x] = np.sum(color, axis=0)

    print(f"Mean absolute error: {np.nanmean(video_mae):.3f}")
    vmin = np.percentile(video_mae[~np.isnan(video_mae)], 1)
    vmax = np.percentile(video_mae[~np.isnan(video_mae)], 99)
    video_mae_vis = pa.applyColorMap(video_mae, "inferno", vmin=vmin, vmax=vmax)
    cv2.imwrite("mae.png", video_mae_vis[0])

    # exit()

    # Extract non-zero elements and print
    print(video_color[video_color != 0])

    cv2.imwrite("color.png", np.clip(video_color[0] * 255, 0, 255).astype(np.uint8))

    print("Visualize")
    img_mueller_vis_list = [ee.mueller_image(video_mm[i]) for i in range(num_frames)]

    print("Save")
    for i, img_mueller_vis in enumerate(img_mueller_vis_list):
        cv2.imwrite(f"mueller_{i:03d}.png", img_mueller_vis)

    # print("Show")
    # while True:
    #     key = -1
    #     for i, img_mueller_vis in enumerate(img_mueller_vis_list):
    #         img_mueller_vis = cv2.resize(img_mueller_vis, None, fx=0.25, fy=0.25)
    #         cv2.imshow("mueller", img_mueller_vis)
    #         key = cv2.waitKey(30)
    #         if key == ord("q"):
    #             break
    #     if key == ord("q"):
    #         break


if __name__ == "__main__":
    main()
