import numpy as np
import eventellipsometry as ee
import cv2

import polanalyser as pa


def main():
    np.random.seed(0)
    np.set_printoptions(precision=3, suppress=True)
    filename_raw = "recordings/recording_2024-07-30_14-29-23.raw"  # 0, 30Hz
    # filename_raw = "recordings/recording_2024-07-29_21-48-46.raw"  # -10, 30Hz
    # filename_raw = "recordings/recording_2024-07-29_21-51-08.raw"  # -20, 30Hz, i=5 good
    filename_raw = "recordings/recording_2024-07-30_13-32-46.raw"  # -30, 30Hz
    filename_raw = "recordings/recording_2024-07-30_13-39-26.raw"  # -30, 48Hz
    # filename_raw = "recordings/recording_2024-07-30_13-43-17.raw"  # -30, 20Hz
    # filename_raw = "recordings/recording_2024-07-30_15-45-54.raw"  # -20, 30Hz
    filename_raw = "recordings/recording_2024-07-30_16-06-55.raw"  # -20, 40Hz
    filename_raw = "recordings/recording_2024-07-30_16-42-37.raw"  # -20, 40Hz, weak light
    filename_raw = "recordings/recording_2024-07-30_19-08-37.raw"  # -20, 30Hz, linear polarizer +90
    # filename_raw = "recordings/recording_2024-07-30_18-54-31.raw"  # -20, 40Hz, linear polarizer +0
    # filename_raw = "recordings/recording_2024-07-31_13-55-06.raw"  # -20, 30Hz, air, 1
    # filename_raw = "recordings/recording_2024-07-31_13-56-35.raw"  # -20, 30Hz, air, 2, bounary
    # filename_raw = "recordings/recording_2024-07-31_13-57-14.raw"  # -20, 30Hz, air, 3
    # filename_raw = "recordings/recording_2024-07-31_13-57-45.raw"  # -20, 30Hz, air, 4
    # filename_raw = "recordings/recording_2024-07-31_16-12-40.raw"  # -20, 30Hz, air, 5
    # filename_raw = "recordings/recording_2024-07-31_16-13-27.raw"  # -20, 30Hz, air, 6
    # filename_raw = "recordings/recording_2024-07-31_16-15-32.raw"  # -20, 30Hz, air, 7
    # filename_raw = "recordings/recording_2024-07-31_16-15-59.raw"  # -20, 30Hz, air, 8
    # filename_raw = "recordings/recording_2024-07-31_16-17-01.raw"  # -20, 30Hz, air, 9
    # filename_raw = "recordings/recording_2024-07-31_16-17-49.raw"  # -20, 30Hz, air, 10

    filename_raw = "recordings/recording_2024-08-05_20-00-19.raw"  # -20, 30Hz, air1, 100x100
    filename_raw = "recordings/recording_2024-08-05_20-01-21.raw"  # -20, 30Hz, air2, 100x100
    filename_raw = "recordings/recording_2024-08-05_20-01-33.raw"  # -20, 30Hz, air3, 100x100
    filename_raw = "recordings/recording_2024-08-05_20-02-26.raw"  # -20, 30Hz, air1, 50x50
    filename_raw = "recordings/recording_2024-08-05_20-03-08.raw"  # -20, 30Hz, air2, 50x50
    filename_raw = "recordings/recording_2024-08-05_20-03-40.raw"  # -20, 30Hz, air3, 50x50
    filename_raw = "recordings/recording_2024-08-05_20-04-36.raw"  # -20, 30Hz, air1, 25x25
    filename_raw = "recordings/recording_2024-08-05_20-04-59.raw"  # -20, 30Hz, air2, 25x25
    filename_raw = "recordings/recording_2024-08-05_20-05-49.raw"  # -20, 30Hz, air3, 25x25

    filename_raw = "recordings/recording_2024-08-05_20-09-14.raw"  # -20, 30Hz, polarizer0, 100x100
    filename_raw = "recordings/recording_2024-08-05_20-08-40.raw"  # -20, 30Hz, polarizer0, 50x50
    filename_raw = "recordings/recording_2024-08-05_20-08-07.raw"  # -20, 30Hz, polarizer0, 25x25

    filename_raw = "recordings/recording_2024-08-05_20-10-39.raw"  # -20, 30Hz, polarizer90, 100x100
    filename_raw = "recordings/recording_2024-08-05_20-11-01.raw"  # -20, 30Hz, polarizer90, 50x50
    filename_raw = "recordings/recording_2024-08-05_20-11-46.raw"  # -20, 30Hz, polarizer90, 25x25

    filename_raw = "recordings/recording_2024-08-21_20-09-27.raw"  # -20, 30Hz, air, 100x100

    filename_raw = "recordings/recording_2024-08-25_15-33-02.raw"  # -30, 30Hz, air2, 100x100

    filename_raw = "recordings/recording_2024-08-25_20-49-55.raw"  # -20, 30Hz, air, 540nm, 100x100, 5th

    ee.imwrite_mueller("air.png", np.eye(4))
    ee.imwrite_mueller("plarizer0.png", pa.polarizer(np.deg2rad(0)))
    ee.imwrite_mueller("plarizer90.png", pa.polarizer(np.deg2rad(90)))

    events = ee.read_event(filename_raw)
    print(events.keys())
    bias_diff_on = events["bias_diff_off"]
    thresholds = {-30: 0.08, -20: 0.15, -10: 0.18, 0: 0.27}
    C = thresholds[bias_diff_on]
    print(f"Threshold C: {C}")

    events_fast = ee.FastEventAccess(events)

    # Triggers
    trig_t = events["trig_t"]
    trig_p = events["trig_p"]
    trig_t_x1 = trig_t[trig_p == 0]
    trig_t_x5 = trig_t[trig_p == 1]

    diff = trig_t_x5[0] - trig_t_x1[0]
    if diff < 0:
        # Delete first element of trig_t_x5
        trig_t_x5 = trig_t_x5[1:]
        diff = trig_t_x5[0] - trig_t_x1[0]

    if len(trig_t_x5) > len(trig_t_x1):
        trig_t_x5 = trig_t_x5[:-1]

    if len(trig_t_x5) < len(trig_t_x1):
        trig_t_x1 = trig_t_x1[:-1]

    period = np.diff(trig_t_x1)
    period = np.append(period, period[-1])  # Add last element
    trig_t_diff = trig_t_x5 - trig_t_x1  # [0, period / 5]
    phi_offsets = trig_t_diff / period * np.pi

    print("Triggers:", len(trig_t_x1), len(trig_t_x5))
    print(f"Avg. frequency: {1 / np.mean(period) * 1000000:.2f} fps")
    for i, (trig_t_x1_i, trig_t_x5_i, trig_t_diff_i, period_i, phi_offset_i) in enumerate(zip(trig_t_x1, trig_t_x5, trig_t_diff, period, phi_offsets)):
        print(f"Triggers {i}: {trig_t_x1_i:d}, {trig_t_x5_i:d}, {trig_t_diff_i:d}, {period_i:d}, {np.rad2deg(phi_offset_i):.2f}")

    # plt.plot(trig_t_x1, np.rad2deg(phi_offset), "o")
    # plt.show()

    width = events["width"]
    height = events["height"]

    img_loss_list = []
    for frame_i in range(10):
        print("-------------------------")
        print(f"Frame {frame_i}")
        t_min = trig_t_x1[frame_i]
        t_max = trig_t_x1[frame_i + 1]
        phi_offset = phi_offsets[frame_i].item()
        t, p = events_fast.get(width // 2, height // 2, t_min, t_max)
        print(f"Number of events (t_min, t_max): {len(t)} ({t_min}, {t_max})")

        print(f"Minimum time difference: {np.min(np.diff(t))}")
        # t_refr = 10
        t = (t - t_min) / (t_max - t_min) * np.pi

        t_diff = np.convolve(t, [1, -1], mode="valid")

        # t_diff = t_diff - (t_refr / (t_max - t_min) * np.pi)

        t = np.convolve(t, [0.5, 0.5], mode="valid")
        p = (2 * p - 1)[1:]
        gt = 1 / t_diff * p * C

        # gt = gt * ((np.abs(gt) + 0.0001) / np.abs(gt))

        # Calib
        dop = 0.6
        h_loss = 360
        w_loss = 360
        img_loss = np.zeros((h_loss, w_loss))
        phi1_candidates = np.linspace(0, np.pi, h_loss, endpoint=False)
        phi2_candidates = np.linspace(0, np.pi, w_loss, endpoint=False)
        for j, phi1_i in enumerate(phi1_candidates):
            # for i, phi2_i in enumerate(np.linspace(phi_offset, phi_offset + np.pi, w_loss)):
            for i, phi2_i in enumerate(phi2_candidates):
                # phi2_offset = (phi2_i - phi_offset) % (2 * np.pi / 1)
                M = np.diag([1, dop, dop, dop]).astype(np.float32)
                pred = ee.diffLn(M.flatten(), t.astype(np.float32), phi1_i, phi2_i - phi_offset * 5)
                loss = np.sum(np.abs(gt - pred) ** 2)
                img_loss[j, i] = loss

        img_loss_list.append(img_loss)

        img_loss_vis = pa.applyColorMap(img_loss, "jet", vmin=np.percentile(img_loss, 1), vmax=np.percentile(img_loss, 99))
        cv2.imwrite(f"loss-{frame_i}.png", img_loss_vis)

    img_loss = np.mean(img_loss_list, axis=0)
    img_loss_vis = pa.applyColorMap(img_loss, "jet", vmin=np.percentile(img_loss, 1), vmax=np.percentile(img_loss, 99))
    cv2.imwrite("loss-mean.png", img_loss_vis)
    index = np.unravel_index(np.argmin(img_loss), img_loss.shape)
    phi1 = phi1_candidates[index[0]]
    phi2 = phi2_candidates[index[1]]
    print("Calibration")
    print(f"loss (mean): {np.mean(img_loss):.2f}")
    print(f"loss (min): {np.min(img_loss):.2f}")
    print(f"phi1: {np.rad2deg(phi1):.2f}")
    print(f"phi2: {np.rad2deg(phi2):.2f}")


if __name__ == "__main__":
    main()
