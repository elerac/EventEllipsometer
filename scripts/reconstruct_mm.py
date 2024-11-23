import argparse
import time
from pathlib import Path
import numpy as np
import eventellipsometry as ee
from recordings.filenames import filename_raw
import calib


def main():
    np.set_printoptions(precision=3, suppress=True)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--filename_raw", type=Path, default=filename_raw)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--max_iter_svd", "--iter_svd", type=int, default=5)
    parser.add_argument("--tol", type=float, default=0.01)
    parser.add_argument("--max_iter_propagate", "--iter_propagate", type=int, default=10)
    args = parser.parse_args()

    print("Read event data")
    print(f"  {args.filename_raw}")
    events = ee.read_event(args.filename_raw)
    width = events["width"]
    height = events["height"]

    mode = events.get("mode", None)
    print(f"  Mode: {mode}")

    if mode == "synthetic":
        C = float(events["thresh"])
        img_C_on = C
        img_C_off = -C
        phi1 = 0
        phi2 = 0
        t_refr = 0
        print("Thresholds")
        print(f"  Con : {C:.3f}")
        print(f"  Coff: {-C:.3f}")
    else:
        bias_diff = int(events["bias_diff"])
        bias_diff_on = int(events["bias_diff_on"])
        bias_diff_off = int(events["bias_diff_off"])
        bias_fo = int(events["bias_fo"])
        bias_hpf = int(events["bias_hpf"])
        bias_refr = int(events["bias_refr"])
        print(f"  Bias diff: {bias_diff}")
        print(f"  Bias diff on: {bias_diff_on}")
        print(f"  Bias diff off: {bias_diff_off}")
        print(f"  Bias fo: {bias_fo}")
        print(f"  Bias hpf: {bias_hpf}")
        print(f"  Bias refr: {bias_refr}")

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
        for i, (trig_t_x1_i, trig_t_x5_i, trig_t_diff_i, period_i, phi_offset_i) in enumerate(zip(trig_t_x1, trig_t_x5, trig_t_diff, period, phi_offsets)):
            print(f"Triggers {i}: {trig_t_x1_i:d}, {trig_t_x5_i:d}, {trig_t_diff_i:d}, {period_i:d}, {np.rad2deg(phi_offset_i):.2f}")

        # Calib
        print("Calibration parameters")
        print(f"  QWP offset")
        phi1 = calib.phi1
        phi2 = calib.phi2
        print(f"    phi1: {np.rad2deg(phi1):.2f} ({phi1:.2f})")
        print(f"    phi2: {np.rad2deg(phi2):.2f} ({phi2:.2f})")
        print(f"  Thresholds")
        print(f"    Con : {bias_diff=}, {bias_diff_on=}")
        print(f"    Coff: {bias_diff=}, {bias_diff_off=}")
        img_C_on = calib.thresh_on(bias_diff, bias_diff_on)
        img_C_off = calib.thresh_off(bias_diff, bias_diff_off)
        img_C_on = np.clip(img_C_on, 1e-9, None)
        img_C_off = np.clip(img_C_off, None, -1e-9)
        t_refr = calib.refractory_period(bias_diff, bias_diff_on, bias_diff_off, bias_fo, bias_hpf, bias_refr)
        print(f"  Refractory period: {t_refr} us")

    print("Construct dataframes")
    ellipsometry_eventmaps = ee.construct_dataframes(events["x"], events["y"], events["t"], events["p"], width, height, events["trig_t"], events["trig_p"], img_C_on, img_C_off, t_refr)
    if args.max_frames is not None:
        ellipsometry_eventmaps = ellipsometry_eventmaps[: args.max_frames]
    print(f"  Number of frames: {len(ellipsometry_eventmaps)}")

    # ellipsometry_eventmaps = ee.add_noise(ellipsometry_eventmaps)

    print("Start reconstruction")
    time_start = time.time()
    video_mm = ee.fit_mueller(ellipsometry_eventmaps, phi1, phi2, max_iter_svd=args.max_iter_svd, tol=args.tol, max_iter_propagate=args.max_iter_propagate, verbose=True)
    video_mm = np.array(video_mm)
    print(f"Total reconstruction time: {time.time() - time_start:.2f} s")
    num_frames, height, width, _, _ = video_mm.shape

    print("Save Mueller matrix video")
    dir_dst = Path(f"results/{args.filename_raw.stem}")
    dir_dst.mkdir(exist_ok=True, parents=True)
    filename_dst_stem = f"{args.filename_raw.stem}_iter_svd{args.max_iter_svd}_tol{args.tol}_iter_propagate{args.max_iter_propagate}"

    filename_npy = dir_dst / f"{filename_dst_stem}.npy"
    np.save(filename_npy, video_mm)
    print(f"  npy: {filename_npy}")

    ee.visualize_mueller_video(filename_npy)


if __name__ == "__main__":
    main()
