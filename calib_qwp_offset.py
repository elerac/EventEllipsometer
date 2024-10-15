import numpy as np
import eventellipsometry as ee
import cv2
import matplotlib.pyplot as plt
import polanalyser as pa
import os


def main():
    np.random.seed(0)
    np.set_printoptions(precision=3, suppress=True)
    filename_raw = "./recordings/calib_qwp_offset/recording_2024-10-15_16-58-31.raw"  # QWP(pi/4)

    events = ee.read_event(filename_raw)
    bias_diff = int(events["bias_diff"])
    bias_diff_on = int(events["bias_diff_on"])
    bias_diff_off = int(events["bias_diff_off"])
    print(f"Bias diff: {bias_diff}")
    print(f"Bias diff on: {bias_diff_on}")
    print(f"Bias diff off: {bias_diff_off}")
    width = events["width"]
    height = events["height"]
    filename_C_on = f"calib/thresh/on/diff{bias_diff}_diff_on{bias_diff_on}_diff_off{bias_diff_off}.npy"
    filename_C_off = f"calib/thresh/off/diff{bias_diff}_diff_on{bias_diff_on}_diff_off{bias_diff_off}.npy"
    img_C_on = np.load(filename_C_on).astype(np.float32)
    img_C_off = np.load(filename_C_off).astype(np.float32)

    ellipsometry_eventmaps = ee.construct_dataframes(events["x"], events["y"], events["t"], events["p"], width, height, events["trig_t"], events["trig_p"], img_C_on, img_C_off)

    x = width // 2
    y = height // 2
    theta, dlogI, weights, phi_offset = ellipsometry_eventmaps[0].get(x, y)

    h_error = 360
    w_error = 360
    img_error = np.zeros((h_error, w_error))
    phi1_candidates = np.linspace(0, np.pi, h_error, endpoint=False)
    phi2_candidates = np.linspace(0, np.pi, w_error, endpoint=False)
    M_ref = pa.depolarizer(0.95) @ pa.qwp(np.pi / 4) @ pa.depolarizer(0.95)
    for j, phi1 in enumerate(phi1_candidates):
        for i, phi2 in enumerate(phi2_candidates):
            n_coffs, d_coffs = ee.calculate_ndcoffs(theta, phi1, phi2 - phi_offset * 5)
            dlogI_pred = (n_coffs @ M_ref.flatten()) / (d_coffs @ M_ref.flatten())
            error = np.sum(np.abs(dlogI - dlogI_pred) ** 2)
            img_error[j, i] = error

    # Get the minimum error
    index = np.unravel_index(np.argmin(img_error), img_error.shape)
    phi1 = phi1_candidates[index[0]]
    phi2 = phi2_candidates[index[1]]
    print("Calibration")
    print(f"error (min): {np.min(img_error):.2f}")
    print(f"phi1: {np.rad2deg(phi1):.2f}")
    print(f"phi2: {np.rad2deg(phi2):.2f}")

    # Write the result to calib/qwp_offset.txt
    filename_txt = "calib/qwp_offset/qwp_offset.txt"
    os.makedirs(os.path.dirname(filename_txt), exist_ok=True)
    with open(filename_txt, "w") as f:
        f.write(f"{filename_raw} % filename_raw\n")
        f.write(f"{phi1} % phi1\n")
        f.write(f"{phi2} % phi2\n")

    # Visualize the error map
    vmin = np.percentile(img_error, 1)
    vmax = np.percentile(img_error, 99)
    print(f"vmin: {vmin}, vmax: {vmax}")
    img_error_vis = pa.applyColorMap(img_error, "jet", vmin=vmin, vmax=vmax)
    filename_png = "calib/qwp_offset/qwp_offset_error.png"
    cv2.imwrite(filename_png, img_error_vis)

    # Plot again
    n_coffs, d_coffs = ee.calculate_ndcoffs(theta, phi1, phi2 - phi_offset * 5)
    dlogI_pred = (n_coffs @ M_ref.flatten()) / (d_coffs @ M_ref.flatten())
    plt.plot(theta, dlogI, "o", label="data")
    plt.plot(theta, dlogI_pred, label="pred")
    plt.xlim(0, np.pi)
    vmax = np.max(np.abs(dlogI))
    plt.ylim(-vmax * 1.05, vmax * 1.05)
    filename_pdf = "calib/qwp_offset/qwp_offset_plot.pdf"
    plt.savefig(filename_pdf, bbox_inches="tight", dpi=500)


if __name__ == "__main__":
    main()
