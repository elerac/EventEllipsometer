import numpy as np
from pathlib import Path
import eventellipsometry as ee
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import trange
import cv2
import polanalyser as pa


# Fit y=a*x+b
def fit(x, y):
    """Fit y=a*x+b"""
    A = np.vstack([x, np.ones(len(x))]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return a, b


def calib_threshold(events, fast_events, ix, iy):
    trig_t = events["trig_t"]
    trig_p = events["trig_p"]
    trig_p_pos = trig_p[trig_p == 1]
    trig_t_pos = trig_t[trig_p == 1]
    trig_p_neg = trig_p[trig_p == 0]
    trig_t_neg = trig_t[trig_p == 0]

    if trig_t_neg[0] < trig_t_pos[0]:
        trig_t_neg = trig_t_neg[1:]

    if len(trig_t_pos) > len(trig_t_neg):
        trig_t_pos = trig_t_pos[:-1]

    t_diff_list = []
    p_list = []
    t_list = []
    valid_list = []
    has_event = False
    for i in range(len(trig_t_pos)):
        t_min = trig_t_pos[i]
        t_max = trig_t_neg[i]
        t, p = fast_events.get(ix, iy, t_min=t_min, t_max=t_max)

        if len(t) < 2:
            continue

        t = t - t_min

        t_norm = t / (t_max - t_min)
        valid = np.logical_and(0.05 < t_norm, t_norm < 0.95)
        # t = t[valid]
        # p = p[valid]

        t_diff = np.convolve(t, [1, -1], mode="valid")
        t = np.convolve(t, [0.5, 0.5], mode="valid")
        p = (2 * p - 1)[1:]
        valid = valid[1:]

        t_diff_list.append(t_diff)
        p_list.append(p)
        t_list.append(t)
        valid_list.append(valid)

        has_event = True

    if not has_event:
        return np.nan, np.nan, np.array([]), np.array([]), np.array([]), np.array([])

    t_diff = np.concatenate(t_diff_list)
    p = np.concatenate(p_list)
    t = np.concatenate(t_list)
    valid = np.concatenate(valid_list)

    a, b = fit(t[valid], t_diff[valid])

    return a, b, t, t_diff, p, valid


def main():
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_index = 0

    filename_raw = "recordings/recording_2024-09-03_15-21-01.raw"  # Ramp down, 10Hz
    # filename_raw = "recordings/recording_2024-09-03_15-29-33.raw"  # Ramp up, 10Hz
    # filename_raw = "recordings/recording_2024-09-03_16-03-19.raw"  # Ramp up, 10Hz, 10x10
    # filename_raw = "recordings/recording_2024-09-03_16-03-48.raw"  # Ramp down, 10Hz, 10x10
    filename_raw = "recordings/recording_2024-09-03_16-38-45.raw"  # Ramp up, 10Hz, 10x10, tailing effect
    # filename_raw = "recordings/recording_2024-09-03_16-56-38.raw"  # Ramp down, 1Hz, 10x10
    # filename_raw = "recordings/recording_2024-09-03_16-58-55.raw"  # Ramp down, 0.5Hz, 10x10
    filename_raw = "recordings/recording_2024-09-03_17-07-03.raw"  # Ramp down, 10Hz, 10x10, 60sec
    filename_raw = "recordings/recording_2024-09-03_17-09-33.raw"  # Ramp up, 10Hz, 10x10, 60sec
    filename_raw = "recordings/recording_2024-09-03_18-49-15.raw"  # Ramp up, 100Hz, 10x10, 60sec
    # filename_raw = "recordings/recording_2024-09-03_18-50-23.raw"  # Ramp down, 100Hz, 10x10, 60sec
    # filename_raw = "recordings/recording_2024-09-03_18-55-14.raw"  # Ramp down, 10Hz, 100x100, 60sec
    # filename_raw = "recordings/recording_2024-09-03_18-58-56.raw"  # Ramp down, 10Hz, All, 30sec
    filename_raw = "recordings/recording_2024-09-03_21-22-32.raw"  # Ramp up, 0.1sec, 10x10, 10sec, Auto
    filename_raw = "recordings/recording_2024-09-03_22-00-45.raw"  # Ramp down, 0.1sec, 10x10, 10sec, Auto
    filename_raw = "recordings/recording_2024-09-04_13-40-57.raw"  # Ramp up, 0.1sec, 10x10, 10sec, Auto
    # filename_raw = "recordings/recording_2024-09-04_13-45-29.raw"  # Ramp down, 0.1sec, All, 10sec, Auto
    filename_raw = "recordings/recording_2024-09-04_13-50-21.raw"  # Ramp down, 0.1sec, All, 15sec, Auto
    filename_raw = "recordings/recording_2024-09-04_13-56-57.raw"  # Ramp down, 0.1sec, 10x10, 60sec, Auto
    # filename_raw = "recordings/recording_2024-09-04_14-07-43.raw"  # Ramp down, 0.1sec, All, 60sec, Auto
    # filename_raw = "recordings/recording_2024-09-04_14-24-11.raw"  # Ramp down, 0.1sec, All, 30sec, Auto
    # filename_raw = "recordings/recording_2024-09-04_15-35-15.raw"  # Ramp up, 0.1sec, 80x80, 10sec, Auto
    # filename_raw = "recordings/recording_2024-09-04_16-53-21.raw"  # Ramp up, 0.1sec, 80x80, 10sec, Auto, on 0, off -20
    # filename_raw = "recordings/recording_2024-09-04_16-54-30.raw"  # Ramp up, 0.1sec, 80x80, 10sec, Auto, on 0, off -35
    filename_raw = "recordings/recording_2024-09-04_17-02-18.raw"  # Ramp down, 0.1sec, 80x80, 10sec, Auto, on 0, off 0
    filename_raw = "recordings/recording_2024-09-04_17-03-30.raw"  # Ramp down, 0.1sec, 80x80, 10sec, Auto, on 0, off -20
    filename_raw = "recordings/recording_2024-09-04_16-55-55.raw"  # Ramp down, 0.1sec, 80x80, 10sec, Auto, on 0, off -35
    filename_raw = "recordings/recording_2024-09-04_17-04-56.raw"  # Ramp down, 0.1sec, 80x80, 10sec, Auto, diff 10, on 0, off -35
    filename_raw = "recordings/recording_2024-09-04_17-06-08.raw"  # Ramp down, 0.1sec, 80x80, 10sec, Auto, diff 20, on 0, off -35

    filename_raw = "recordings/recording_2024-09-04_20-09-28.raw"  # Ramp down, 0.1sec, 10x10, 15sec, Auto, diff 0, on 0, off 0
    # filename_raw = "recordings/recording_2024-09-04_20-10-08.raw"  # Ramp down, 0.1sec, 10x10, 15sec, Auto, diff 10, on 0, off 0
    # filename_raw = "recordings/recording_2024-09-04_20-10-55.raw"  # Ramp down, 0.1sec, 10x10, 15sec, Auto, diff -10, on 0, off 0
    # filename_raw = "recordings/recording_2024-09-04_20-11-33.raw"  # Ramp up, 0.1sec, 10x10, 15sec, Auto, diff 0, on 0, off 0
    # filename_raw = "recordings/recording_2024-09-04_20-12-45.raw"  # Ramp up, 0.1sec, 10x10, 15sec, Auto, diff 10, on 0, off 0
    # filename_raw = "recordings/recording_2024-09-04_20-13-33.raw"  # Ramp up, 0.1sec, 10x10, 15sec, Auto, diff -10, on 0, off 0

    # print(f"Read events from {filename_raw}")

    # events = ee.read_event(filename_raw)
    # fast_events = ee.EventMapPy(events)

    # bias_diff = events["bias_diff"]
    # bias_diff_on = events["bias_diff_on"]
    # bias_diff_off = events["bias_diff_off"]

    # width = events["width"]
    # height = events["height"]

    # for ix in range(width // 2 - 1, width // 2 + 2):
    #     for iy in range(height // 2 - 1, height // 2 + 2):
    #         a, b, t, t_diff, p, valid = calib_threshold(events, fast_events, ix, iy)

    #         color = color_cycle[color_index % len(color_cycle)]
    #         color_index += 1

    #         valid_on = np.bitwise_and(valid, p == 1)
    #         plt.plot(t[valid_on], t_diff[valid_on], marker="o", linestyle="None", alpha=0.5, color=color, markeredgewidth=0, label=f"({ix}, {iy}), C={a:.2f}")
    #         valid_off = np.bitwise_and(valid, p == -1)
    #         plt.plot(t[valid_off], t_diff[valid_off], marker="*", linestyle="None", alpha=0.5, color=color, markeredgewidth=0)
    #         invalid_on = np.bitwise_and(~valid, p == 1)
    #         plt.plot(t[invalid_on], t_diff[invalid_on], marker="o", linestyle="None", alpha=0.1, color=color, markeredgewidth=0)
    #         invalid_off = np.bitwise_and(~valid, p == -1)
    #         plt.plot(t[invalid_off], t_diff[invalid_off], marker="*", linestyle="None", alpha=0.1, color=color, markeredgewidth=0)
    #         plt.plot(t, a * t + b, color=color)

    # plt.xlim(0, 0.1 * 1e6)
    # plt.grid()
    # plt.legend()
    # plt.show()

    calib_rampdown_dir = "recordings/calib_thresh_rampdown"
    param_dirs = [d for d in Path(calib_rampdown_dir).iterdir() if d.is_dir()]

    for param_dir in param_dirs:
        filename_raw_list = list(param_dir.glob("*.raw"))

        bias_diff = ee.utils.get_num_after_keyword(param_dir, "diff")
        bias_diff_on = ee.utils.get_num_after_keyword(param_dir, "diff_on")
        bias_diff_off = ee.utils.get_num_after_keyword(param_dir, "diff_off")
        if not (bias_diff == 0 and bias_diff_on == -35 and bias_diff_off == -35):
            continue
        img_C = np.full((720, 1280), np.nan)
        for filename_raw in filename_raw_list[::8]:
            print(filename_raw)
            roi_x = int(ee.utils.get_num_after_keyword(filename_raw, "x"))
            roi_y = int(ee.utils.get_num_after_keyword(filename_raw, "y"))
            roi_w = int(ee.utils.get_num_after_keyword(filename_raw, "w"))
            roi_h = int(ee.utils.get_num_after_keyword(filename_raw, "h"))

            events = ee.read_event(filename_raw)
            fast_events = ee.EventMap(events["x"], events["y"], events["t"], events["p"], events["width"], events["height"])

            for ix in trange(roi_x, roi_x + roi_w):
                for iy in range(roi_y, roi_y + roi_h):
                    a, b, t, t_diff, p, valid = calib_threshold(events, fast_events, ix, iy)
                    img_C[iy, ix] = a

            filename_npy = f"calibration/Coff_diff{bias_diff}_diff_off{bias_diff_off}.npy"
            Path(filename_npy).parent.mkdir(parents=True, exist_ok=True)
            np.save(filename_npy, img_C)

            vmin = np.percentile(img_C[~np.isnan(img_C)], 1)
            vmax = np.percentile(img_C[~np.isnan(img_C)], 99)
            print(vmin, vmax)
            img_C_vis = pa.applyColorMap(img_C, "viridis", vmin=vmin, vmax=vmax)

            cv2.imshow("C", img_C_vis)
            key = cv2.waitKey(100)
            if key == ord("q"):
                break

        # cv2.waitKey(0)


if __name__ == "__main__":
    main()
