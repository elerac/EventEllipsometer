import numpy as np
import eventellipsometer as ee
import cv2
import matplotlib.pyplot as plt
import polanalyser as pa
import os
import datetime


def main():
    np.random.seed(0)
    np.set_printoptions(precision=3, suppress=True)
    # filename_raw = "./recordings/calib_qwp_offset/recording_2024-10-19_17-17-53.raw"  # QWP(pi/4)
    filename_raw = "./recordings/calib_qwp_offset/recording_2024-10-20_17-41-08.raw"  # QWP, 200x200

    print(f"Reading {filename_raw}")

    events = ee.read_event(filename_raw)
    bias_diff = int(events["bias_diff"])
    bias_diff_on = int(events["bias_diff_on"])
    bias_diff_off = int(events["bias_diff_off"])
    bias_fo = int(events["bias_fo"])
    bias_hpf = int(events["bias_hpf"])
    bias_refr = int(events["bias_refr"])
    print(f"Bias diff: {bias_diff}")
    print(f"Bias diff on: {bias_diff_on}")
    print(f"Bias diff off: {bias_diff_off}")
    print(f"Bias fo: {bias_fo}")
    print(f"Bias hpf: {bias_hpf}")
    print(f"Bias refr: {bias_refr}")
    width = events["width"]
    height = events["height"]
    filename_C_on = f"calib/thresh/on/diff{bias_diff}_diff_on{bias_diff_on}_diff_off{bias_diff_off}.npy"
    filename_C_off = f"calib/thresh/off/diff{bias_diff}_diff_on{bias_diff_on}_diff_off{bias_diff_off}.npy"
    img_C_on = np.load(filename_C_on).astype(np.float32)
    img_C_off = np.load(filename_C_off).astype(np.float32)
    img_C_on = np.clip(img_C_on, 1e-9, None)
    img_C_off = np.clip(img_C_off, None, -1e-9)

    ellipsometry_eventmaps = ee.construct_dataframes(events["x"], events["y"], events["t"], events["p"], width, height, events["trig_t"], events["trig_p"], img_C_on, img_C_off)

    x = width // 2
    y = height // 2
    theta, dlogI, weights, phi_offset = ellipsometry_eventmaps[1].get(x, y)

    h_error = 360
    w_error = 360
    img_error = np.zeros((h_error, w_error))
    phi1_candidates = np.linspace(0, np.pi, h_error, endpoint=False)
    phi2_candidates = np.linspace(0, np.pi, w_error, endpoint=False)
    M_ref = pa.depolarizer(0.8) @ pa.qwp(np.pi / 4)
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
    print(f"phi1: {np.rad2deg(phi1):.2f} ({phi1:.3f})")
    print(f"phi2: {np.rad2deg(phi2):.2f} ({phi2:.3f})")
    print(f"phi1 + pi/2: {np.rad2deg(phi1 + np.pi/2) % 180:.2f} ({(phi1 + np.pi/2) % np.pi:.3f})")
    print(f"phi2 + pi/2: {np.rad2deg(phi2 + np.pi/2) % 180:.2f} ({(phi2 + np.pi/2) % np.pi:.3f})")

    # Write the result to calib/qwp_offset.txt
    datetime_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename_phi1 = "calib/qwp_offset/phi1.txt"
    os.makedirs(os.path.dirname(filename_phi1), exist_ok=True)
    with open(filename_phi1, "w") as f:
        f.write(f"% {datetime_now}\n")
        f.write(f"% {filename_raw}\n")
        f.write(f"{phi1:.6f}")
    filename_phi2 = "calib/qwp_offset/phi2.txt"
    with open(filename_phi2, "w") as f:
        f.write(f"% {datetime_now}\n")
        f.write(f"% {filename_raw}\n")
        f.write(f"{phi2:.6f}")

    # Visualize the error map
    vmin = np.min(img_error)
    vmax = np.max(img_error)
    print(f"vmin: {vmin}, vmax: {vmax}")
    img_error_vis = pa.applyColorMap(img_error, "jet", vmin=vmin, vmax=vmax)
    filename_png = "calib/qwp_offset/qwp_offset_error.png"
    cv2.imwrite(filename_png, img_error_vis)

    # Plot again
    plt.plot(theta, dlogI, "o", label="data")
    theta = np.linspace(0, np.pi, 360).astype(np.float32)
    n_coffs, d_coffs = ee.calculate_ndcoffs(theta, phi1, phi2 - phi_offset * 5)
    dlogI_pred = (n_coffs @ M_ref.flatten()) / (d_coffs @ M_ref.flatten())
    plt.plot(theta, dlogI_pred, label="pred")
    plt.xlim(0, np.pi)
    vmax = np.max(np.abs(dlogI))
    plt.ylim(-vmax * 1.05, vmax * 1.05)
    filename_pdf = "calib/qwp_offset/qwp_offset_plot.pdf"
    plt.savefig(filename_pdf, bbox_inches="tight", dpi=500)

    img_colorbar = np.linspace(0, 1, 256).reshape(1, -1)
    img_colorbar = np.repeat(img_colorbar, 16, axis=0)
    img_colorbar_jet = pa.applyColorMap(img_colorbar, "jet", vmin=0, vmax=1)
    filename_colorbar = "calib/qwp_offset/qwp_offset_colorbar.png"
    cv2.imwrite(filename_colorbar, img_colorbar_jet)


if __name__ == "__main__":
    main()
