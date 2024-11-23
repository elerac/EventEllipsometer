import numpy as np
from pathlib import Path
import eventellipsometry as ee
from pathlib import Path
from tqdm import tqdm
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

        # cut off the first and last 5% of the data
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
    # Parameters
    bias_diff = 20
    bias_diff_on = bias_diff_off = -35
    on_off = "off"
    height = 720
    width = 1280

    if on_off == "on":
        calib_rampdown_dir = "recordings/calib_thresh_rampup"
        filename_npy = f"calib/thresh/on/diff{bias_diff}_diff_on{bias_diff_on}_diff_off{bias_diff_off}.npy"
    else:
        calib_rampdown_dir = "recordings/calib_thresh_rampdown"
        filename_npy = f"calib/thresh/off/diff{bias_diff}_diff_on{bias_diff_on}_diff_off{bias_diff_off}.npy"

    # Search for the calibration data
    param_dirs = [d for d in Path(calib_rampdown_dir).iterdir() if d.is_dir()]
    param_dir = ""
    hit = False
    for i in range(len(param_dirs)):
        param_dir = param_dirs[i]
        keywords = ["diff", "diff_on", "diff_off"]
        _bias_diff, _bias_diff_on, _bias_diff_off = ee.utils.get_num_after_keyword(param_dir, keywords)
        if bias_diff == _bias_diff and bias_diff_on == _bias_diff_on and bias_diff_off == _bias_diff_off:
            hit = True
            break
    if not hit:
        raise ValueError("No calibration data found.")

    print("param_dir:", param_dir)

    # Estimate the contrast threshold
    img_C = np.full((height, width), np.nan)
    filename_raw_list = list(param_dir.glob("*.raw"))
    pbar = tqdm(filename_raw_list)
    for filename_raw in pbar:
        pbar.set_description(f"{filename_raw.stem}")
        roi_x, roi_y, roi_w, roi_h = ee.utils.get_num_after_keyword(filename_raw, ["x", "y", "w", "h"])

        events = ee.read_event(filename_raw)
        t = events["t"]
        index = np.argsort(t)
        events["x"] = events["x"][index]
        events["y"] = events["y"][index]
        events["t"] = events["t"][index]
        events["p"] = events["p"][index]

        event_map = ee.EventMap(events["x"], events["y"], events["t"], events["p"], events["width"], events["height"])

        for ix in range(roi_x, roi_x + roi_w):
            for iy in range(roi_y, roi_y + roi_h):
                a, b, t, t_diff, p, valid = calib_threshold(events, event_map, ix, iy)
                img_C[iy, ix] = a

        # Visualize the contrast map
        vmin = np.percentile(img_C[~np.isnan(img_C)], 1).item()
        vmax = np.percentile(img_C[~np.isnan(img_C)], 99).item()
        img_C_vis = pa.applyColorMap(img_C, "viridis", vmin=vmin, vmax=vmax)
        cv2.imshow("C", img_C_vis)
        key = cv2.waitKey(100)
        if key == ord("q"):
            break

    Path(filename_npy).parent.mkdir(parents=True, exist_ok=True)
    np.save(filename_npy, img_C)
    print(f"Saved: {filename_npy}")
    print(f"average: {np.nanmean(img_C):.3f}")
    print(f"std: {np.nanstd(img_C):.3f}")
    print(f"min: {np.nanmin(img_C):.3f}")
    print(f"max: {np.nanmax(img_C):.3f}")


if __name__ == "__main__":
    main()
