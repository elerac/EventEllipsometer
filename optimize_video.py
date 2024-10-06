from pathlib import Path
import time
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import eventellipsometry as ee

# from scipy.optimize import least_squares
import cv2


from tqdm import trange

import polanalyser as pa
from check_valid_mueller import ismueller, isstokes

I = 1.0j
Tn = 1 / 4 * np.array([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, -I, 0, 0, I, 0], [0, 0, 1, 0, 0, 0, 0, I, 1, 0, 0, 0, 0, -I, 0, 0], [0, 0, 0, 1, 0, 0, -I, 0, 0, I, 0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, I, 0, 0, -I, 0], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1], [0, 0, 0, I, 0, 0, 1, 0, 0, 1, 0, 0, -I, 0, 0, 0], [0, 0, I, 0, 0, 0, 0, 1, -I, 0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, -I, 1, 0, 0, 0, 0, I, 0, 0], [0, 0, 0, -I, 0, 0, 1, 0, 0, 1, 0, 0, I, 0, 0, 0], [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1], [0, I, 0, 0, -I, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, I, 0, 0, -I, 0, 0, 1, 0, 0, 0], [0, 0, -I, 0, 0, 0, 0, 1, I, 0, 0, 0, 0, 1, 0, 0], [0, -I, 0, 0, I, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0], [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1]], dtype=complex)  # first row  # second row  # third row  # fourth row
Tinv = LA.inv(Tn)


def T(x):
    return np.reshape((Tn @ np.reshape(x, (16, 1))), (4, 4))


def invT(x):
    return np.reshape((Tinv @ np.reshape(x, (16, 1))), (4, 4))


# accoring to JoseJ.Gil (On optimal filtering of measured Mueller matrices)
# The definition of normlaized eigenvalues are slightly different
# We need to check.
def GenericFilter(x):
    e, v = LA.eigh(T(x))
    D = np.diag(np.clip(e, 0, None))
    return (invT(v @ D @ LA.inv(v))).real


def GenericM00eigenFilter(y):
    neg_e = y[y <= 0]
    d = 0
    d = np.sum(neg_e) / (4 - np.size(neg_e))
    y = np.clip(y, 0, None)
    y[y != 0] = y[y != 0] + d
    # check for newly produced negative
    # eigenvalues and filter again if necessary
    neg_y = np.clip(y, None, 0)
    s = np.size((neg_y[neg_y != 0]))
    if s != 0:
        y = GenericM00eigenFilter(y)
    # done
    return y


def GenericZfilter(x):
    e, v = LA.eigh(T(x))
    D = GenericM00eigenFilter(e)
    D = np.diag(D)
    return (invT(v @ D @ LA.inv(v))).real


def main():
    np.set_printoptions(precision=3, suppress=True)

    np.random.seed(0)
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
    # filename_raw = "recordings/recording_2024-07-31_13-56- .raw"  # -20, 30Hz, air, 2, bounary
    # filename_raw = "recordings/recording_2024-07-31_13-57-14.raw"  # -20, 30Hz, air, 3
    # filename_raw = "recordings/recording_2024-07-31_13-57-45.raw"  # -20, 30Hz, air, 4
    # filename_raw = "recordings/recording_2024-07-31_16-12-40.raw"  # -20, 30Hz, air, 5
    # filename_raw = "recordings/recording_2024-07-31_16-13-27.raw"  # -20, 30Hz, air, 6
    # filename_raw = "recordings/recording_2024-07-31_16-15-32.raw"  # -20, 30Hz, air, 7
    # filename_raw = "recordings/recording_2024-07-31_16-15-59.raw"  # -20, 30Hz, air, 8
    # filename_raw = "recordings/recording_2024-07-31_16-17-01.raw"  # -20, 30Hz, air, 9
    # filename_raw = "recordings/recording_2024-07-31_16-17-49.raw"  # -20, 30Hz, air, 10

    filename_raw = "recordings/recording_2024-08-05_20-00-19.raw"  # -20, 30Hz, air1, 100x100
    # filename_raw = "recordings/recording_2024-08-05_20-01-21.raw"  # -20, 30Hz, air2, 100x100
    # filename_raw = "recordings/recording_2024-08-05_20-01-33.raw"  # -20, 30Hz, air3, 100x100
    # filename_raw = "recordings/recording_2024-08-05_20-02-26.raw"  # -20, 30Hz, air1, 50x50
    # filename_raw = "recordings/recording_2024-08-05_20-03-08.raw"  # -20, 30Hz, air2, 50x50
    # filename_raw = "recordings/recording_2024-08-05_20-03-40.raw"  # -20, 30Hz, air3, 50x50
    # filename_raw = "recordings/recording_2024-08-05_20-04-36.raw"  # -20, 30Hz, air1, 25x25
    # filename_raw = "recordings/recording_2024-08-05_20-04-59.raw"  # -20, 30Hz, air2, 25x25
    # filename_raw = "recordings/recording_2024-08-05_20-05-49.raw"  # -20, 30Hz, air3, 25x25

    # filename_raw = "recordings/recording_2024-08-05_20-09-14.raw"  # -20, 30Hz, polarizer0, 100x100
    # filename_raw = "recordings/recording_2024-08-05_20-08-40.raw"  # -20, 30Hz, polarizer0, 50x50
    # filename_raw = "recordings/recording_2024-08-05_20-08-07.raw"  # -20, 30Hz, polarizer0, 25x25

    # filename_raw = "recordings/recording_2024-08-05_20-10-39.raw"  # -20, 30Hz, polarizer90, 100x100
    # filename_raw = "recordings/recording_2024-08-05_20-11-01.raw"  # -20, 30Hz, polarizer90, 50x50
    # filename_raw = "recordings/recording_2024-08-05_20-11-46.raw"  # -20, 30Hz, polarizer90, 25x25

    # New calibration
    # -------------------------
    # filename_raw = "recordings/recording_2024-08-21_20-09-27.raw"  # -20, 30Hz, air, 100x100
    # filename_raw = "recordings/recording_2024-08-21_21-25-14.raw"  # -20, 30Hz, polarizer90, 512x512
    # filename_raw = "recordings/recording_2024-08-21_21-40-23.raw"  # -20, 30Hz, polarizer90, 512x512, motor direction of light side has flipped
    # filename_raw = "recordings/recording_2024-08-22_11-08-01.raw"
    # filename_raw = "recordings/recording_2024-08-22_11-25-44.raw"  # -30, 30H, lens
    # filename_raw = "recordings/recording_2024-08-22_12-54-28.raw"  # -30, 30H, metal
    # filename_raw = "recordings/recording_2024-08-22_13-40-15.raw"  # -30, 30H, metal, dark

    # New calibration
    # -------------------------
    # filename_raw = "recordings/recording_2024-08-25_15-33-02.raw"  # -30, 30Hz, air2, 100x100
    # filename_raw = "recordings/recording_2024-08-25_15-47-20.raw"  # -30, 30Hz, air2 (+green filter), 100x100,
    # filename_raw = "recordings/recording_2024-08-25_17-36-09.raw"  # -30, 30Hz, air2 (+green filter), 10x10, 2nd
    # filename_raw = "recordings/recording_2024-08-25_20-13-57.raw"  # -20, 30Hz, air , 100x100
    # filename_raw = "recordings/recording_2024-08-25_20-22-03.raw"  # -20, 30Hz, air (umpbrella), 100x100
    # filename_raw = "recordings/recording_2024-08-25_20-26-24.raw"  # -20, 30Hz, air (umpbrella), 100x100, 2nd
    # filename_raw = "recordings/recording_2024-08-25_20-28-44.raw"  # -20, 30Hz, air (umpbrella), 100x100, 3rd
    # filename_raw = "recordings/recording_2024-08-25_20-30-28.raw"  # -20, 30Hz, air (umpbrella), 100x100, 4th
    # filename_raw = "recordings/recording_2024-08-25_20-49-55.raw"  # -20, 30Hz, air, 540nm, 100x100, 5th
    # filename_raw = "recordings/recording_2024-08-25_21-05-55.raw"  # -30, 30Hz, air, 540nm, 100x100, 6th
    # filename_raw = "recordings/recording_2024-08-25_21-07-26.raw"  # -30, 30Hz, air, 540nm, 100x100, 7th
    # filename_raw = "recordings/recording_2024-08-25_21-31-16.raw"  # -20, 30Hz, air, 540nm, 512x512, 8th
    # filename_raw = "recordings/recording_2024-08-27_21-54-17.raw"  # -20, 30Hz, metal, 100x100, 4th (good)
    filename_raw = "recordings/recording_2024-08-27_22-16-54.raw"  # -30, 30Hz, NVIDIA GPU, 512x512, good

    events = ee.read_event(filename_raw)
    print(filename_raw)
    print(events.keys())
    print("x", events["x"].dtype, events["x"].shape)
    print("y", events["y"].dtype, events["y"].shape)
    print("t", events["t"].dtype, events["t"].shape)
    print("p", events["p"].dtype, events["p"].shape)
    bias_diff_on = events["bias_diff_off"]
    print(f"Bias diff off: {events['bias_diff_off']}")
    print(f"Bias diff on: {events['bias_diff_on']}")
    thresholds = {-30: 0.08, -20: 0.15, -10: 0.18, 0: 0.27}
    C = thresholds[bias_diff_on]
    print(f"Threshold C: {C}")

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

    # events_fast = ee.EventMapPy(events, t_min=trig_t_x1[0], t_max=trig_t_x1[5])
    events_fast = ee.EventMap(events["x"], events["y"], events["t"], events["p"], events["width"], events["height"])

    width = events["width"]
    height = events["height"]

    img_M_opt_DLT_list = []
    img_M_opt_nonlinear_list = []
    img_M_opt_DLT_prop_list = []
    img_M_opt_nonlinear_prop_list = []

    # Calib
    phi1 = np.deg2rad(56)
    phi2 = np.deg2rad(146)

    # phi1 = np.deg2rad(6.5)
    # phi2 = np.deg2rad(64)

    phi1 = np.deg2rad(6.5)
    phi2 = np.deg2rad(62.5)

    phi1 = phi1 + np.pi / 2
    phi2 = phi2 + np.pi / 2
    phi1 = float(phi1)
    phi2 = float(phi2)

    print("Calibration")
    print(f"phi1: {np.rad2deg(phi1):.2f} ({phi1:.2f})")
    print(f"phi2: {np.rad2deg(phi2):.2f} ({phi2:.2f})")

    roi_w = roi_h = 700

    s = time.time()
    ellipsometry_eventmaps = ee.construct_dataframes(events["x"], events["y"], events["t"], events["p"], width, height, trig_t, trig_p, C, -1.8 * C)
    f = time.time()
    print(f"Time (dataframes): {f - s:.2f} s")

    # data_frame = ellipsometry_eventmaps[0]
    # for i in range(3):
    #     for j in range(3):
    #         theta, dlogI, weight, phi_offset = data_frame.get(width // 2 + i, height // 2 + j)
    #         m = ee.fit(theta, dlogI, phi1, phi2 - 5 * phi_offset, max_iter=10, tol=1e-3, delta=1.35, debug=True)

    #         # add outlier noise (10%)
    #         outlier = np.random.rand(len(dlogI)) > 0.9
    #         dlogI[outlier] = np.random.normal(0, 100, np.sum(outlier))

    #         print("Outlier")
    #         print("delta", 1.35)
    #         m = ee.fit(theta, dlogI, phi1, phi2 - 5 * phi_offset, max_iter=10, tol=1e-3, delta=1.35, debug=True)

    #         print("delta", 1.35 * 1000)
    #         m = ee.fit(theta, dlogI, phi1, phi2 - 5 * phi_offset, max_iter=10, tol=1e-3, delta=1.35 * 1000, debug=True)

    # exit()

    # exit()

    print("fit")
    s = time.time()
    video_mueller = ee.fit_mueller(ellipsometry_eventmaps)
    num_frames = len(ellipsometry_eventmaps)
    video_mueller = np.reshape(video_mueller, (num_frames, height, width, 16))
    video_mueller = np.reshape(video_mueller, (num_frames, height, width, 4, 4))
    print(f"Time: {time.time() - s:.2f} s")

    img_mueller_vis_list = [ee.mueller_image(video_mueller[i]) for i in range(num_frames)]

    for i, img_mueller_vis in enumerate(img_mueller_vis_list):
        cv2.imwrite(f"mueller_{i:03d}.png", img_mueller_vis)

    while True:
        key = -1
        for i, img_mueller_vis in enumerate(img_mueller_vis_list):
            img_mueller_vis = cv2.resize(img_mueller_vis, None, fx=0.25, fy=0.25)
            cv2.imshow("mueller", img_mueller_vis)
            key = cv2.waitKey(30)
            if key == ord("q"):
                break
        if key == ord("q"):
            break

    exit()

    for i, data_frame in enumerate(ellipsometry_eventmaps):
        s = time.time()
        img_mueller_flatten = ee.fit_frame(data_frame)
        print(f"{i} Time: {time.time() - s:.2f} s")
        img_mueller = np.reshape(img_mueller_flatten, (height, width, 16))
        img_mueller = np.reshape(img_mueller, (height, width, 4, 4))
        img_mueller_vis = ee.mueller_image(img_mueller)
        img_mueller_vis = cv2.resize(img_mueller_vis, None, fx=0.25, fy=0.25)
        cv2.imshow("mueller", img_mueller_vis)
        key = cv2.waitKey(30)
        if key == ord("q"):
            break

    exit()

    # print("Ellipsometry eventmaps")
    # exit()

    for frame_i in range(0, 5):
        img_mueller_DLT = np.zeros((roi_h, roi_w, 4, 4))
        img_mueller_NLLS = np.zeros((roi_h, roi_w, 4, 4))
        img_loss_DLT = np.zeros((roi_h, roi_w))
        print("-------------------------")
        print(f"Frame {frame_i}")

        for yi in trange(roi_h):
            for xi in range(roi_w):
                # t, p = events_fast.get(width // 2 + xi, height // 2 + yi, t_min, t_max)
                # t = (t - t_min) / (t_max - t_min) * np.pi
                # t_diff = np.convolve(t, [1, -1], mode="valid")
                # # t_diff = t_diff - (t_refr / (t_max - t_min) * np.pi)
                # t = np.convolve(t, [0.5, 0.5], mode="valid")
                # p = (2 * p - 1)[1:]
                # gt = 1 / t_diff * p * C

                t_list = []
                gt_list = []
                for i in range(1):
                    for j in range(1):
                        for k in range(1):
                            t_min = trig_t_x1[frame_i + k]
                            t_max = trig_t_x1[frame_i + k + 1]
                            phi_offset = phi_offsets[frame_i].item()

                            # t, p = temporal_eventmap.event_maps[frame_i](width // 2 + xi - roi_w // 2 + i, height // 2 + yi - roi_h // 2 + j)

                            t, p = events_fast.get(width // 2 + xi - roi_w // 2 + i, height // 2 + yi - roi_h // 2 + j, t_min, t_max)

                            if len(t) <= 16:
                                continue
                            t = (t - t_min) / (t_max - t_min) * np.pi
                            t_diff = np.convolve(t, [1, -1], mode="valid")
                            t_refr = 55  # 10
                            t_refr = 0
                            t_diff = t_diff - (t_refr / (t_max - t_min) * np.pi)
                            t = np.convolve(t, [0.5, 0.5], mode="valid")
                            p = (2 * p - 1)[1:]
                            # gt = 1 / t_diff * p * C
                            C_ = np.where(p == 1, C, C * 1.8)
                            gt = 1 / t_diff * p * C_

                            # remove zero division
                            index = ~np.isinf(gt)
                            t = t[index]
                            gt = gt[index]

                            t_list.append(t)
                            gt_list.append(gt)

                if len(t_list) == 0:
                    continue

                t = np.concatenate(t_list)
                gt = np.concatenate(gt_list)

                # t, gt = ellipsometry_eventmaps[frame_i].get(width // 2 + xi - roi_w // 2, height // 2 + yi - roi_h // 2)
                # remove zero division
                # index = ~np.isinf(gt2)
                # t2 = t2[index]
                # gt2 = gt2[index]

                # print(np.allclose(t, t2, rtol=1e-3, atol=1e-3), np.allclose(gt, gt2, rtol=1e-3, atol=1e-3), gt[:3], gt2[:3])

                if len(t) < 16:
                    continue

                # DLT
                # pn, pd = ee.calcNumenatorDenominatorCoffs(t.astype(np.float32), phi1, phi2 - 5 * phi_offset)
                # A = pn - np.diag(gt) @ pd
                # w = np.ones(len(gt))
                # w = 1 / np.abs(gt)
                # for _ in range(20):
                #     Aw = np.diag(w / np.sum(w)) @ A
                #     m = ee.svdSolve(Aw.astype(np.float32))
                #     M = m.reshape(4, 4)
                #     m = M.flatten()
                #     pred = (pn @ m) / (pd @ m)
                #     w = 1 / (np.abs(gt - pred) + 0.001)

                #     img_m = ee.mueller_image(M)
                #     cv2.imshow("mueller_DLT", img_m)

                #     M_filtered = GenericFilter(M)
                #     img_m_filtered = ee.mueller_image(M_filtered)
                #     cv2.imshow("mueller_DLT_filtered", img_m_filtered)

                #     key = cv2.waitKey(0)
                #     if key == ord("q"):
                #         break

                # print(t.shape, gt.shape)

                m = ee.fit(
                    t.astype(np.float32),
                    gt.astype(np.float32),
                    phi1,
                    phi2 - 5 * phi_offset,
                    max_iter=10,
                    tol=1e-2,
                    delta=1.35,
                    debug=False,
                )

                # m = np.clip(m, -1, 1)
                M = np.reshape(m, (4, 4))

                img_mueller_DLT[yi, xi] = M

                # m = M.flatten()
                # pred = (pn @ m) / (pd @ m)
                # loss = F.huber_loss(torch.tensor(gt), torch.tensor(pred), delta=0.1, reduction="mean").item()
                # img_loss_DLT[yi, xi] = loss

        img_mueller_DLT_vis = ee.mueller_image(img_mueller_DLT)
        if img_mueller_DLT_vis.shape[0] > 512:
            img_mueller_DLT_vis = cv2.resize(img_mueller_DLT_vis, (512, 512))
        cv2.imshow("mueller_DLT", img_mueller_DLT_vis)
        key = cv2.waitKey(30)

        cv2.imwrite(f"mueller_DLT_{frame_i}.png", img_mueller_DLT_vis)

        # filtering
        img_valid = np.zeros((roi_h, roi_w), dtype=bool)
        for yi in range(roi_h):
            for xi in range(roi_w):
                M = img_mueller_DLT[yi, xi]

                img_mueller_DLT[yi, xi] = GenericFilter(M)

                # isvalid = ismueller(img_mueller_DLT[yi, xi])

                # if isvalid:
                #     img_valid[yi, xi] = True
                # else:
                #     print(f"Invalid mueller matrix at ({xi}, {yi})")

        img_mueller_DLT_vis = ee.mueller_image(img_mueller_DLT)
        if img_mueller_DLT_vis.shape[0] > 512:
            img_mueller_DLT_vis = cv2.resize(img_mueller_DLT_vis, (512, 512))
        cv2.imshow("mueller_DLT_filtered", img_mueller_DLT_vis)
        # cv2.imshow("valid", img_valid.astype(np.uint8) * 255)
        key = cv2.waitKey(30)

        # propagate
        img_mueller_base = np.copy(img_mueller_DLT)
        # img_mueller_base[...] = 0.0
        # img_mueller_base[..., 0, 0] = 1.0
        # img_mueller_base[..., 1, 1] = 0.9
        # img_mueller_base[..., 2, 2] = 0.9
        # img_mueller_base[..., 3, 3] = 0.9
        img_mueller_prop = np.copy(img_mueller_base)
        pbar = trange(60, desc="Propagate")
        delta = 1.35 * 10  # * 20

        def huber_loss(gt, pred, delta, weight):
            # weight = None
            # return np.average(np.abs(gt - pred) ** 2, weights=weight)
            return np.average(np.where(np.abs(gt - pred) < delta, 0.5 * (gt - pred) ** 2, delta * (np.abs(gt - pred) - 0.5 * delta)), weights=weight)

        for _ in pbar:
            loss_list = []
            num_updates = 0
            for yi in range(roi_h):
                for xi in range(roi_w):
                    if np.isnan(img_mueller_base[yi, xi]).any():
                        continue
                    is_updated = False
                    t_min = trig_t_x1[frame_i]
                    t_max = trig_t_x1[frame_i + 1]
                    t, p = events_fast.get(width // 2 + xi - roi_w // 2, height // 2 + yi - roi_h // 2, t_min, t_max)
                    if len(t) <= 16:
                        continue
                    t = (t - t_min) / (t_max - t_min) * np.pi
                    t_diff = np.convolve(t, [1, -1], mode="valid")
                    t_diff = t_diff - (t_refr / (t_max - t_min) * np.pi)
                    t = np.convolve(t, [0.5, 0.5], mode="valid")
                    p = (2 * p - 1)[1:]
                    # gt = 1 / t_diff * p * C
                    C_ = np.where(p == 1, C, C * 1.8)
                    gt = 1 / t_diff * p * C_

                    pn, pd = ee.calcNumenatorDenominatorCoffs(t.astype(np.float32), phi1, phi2 - 5 * phi_offset)
                    A = pn - np.diag(gt) @ pd

                    M_best = img_mueller_base[yi, xi]
                    m = M_best.flatten()
                    pred = (pn @ m) / (pd @ m)
                    # loss_best = F.huber_loss(torch.tensor(gt), torch.tensor(pred), delta=delta).item()
                    loss_best = huber_loss(gt, pred, delta, 1 / np.abs(gt))

                    scheme_speed = [(0, 1), (0, -1), (1, 0), (-1, 0), (5, 0), (-5, 0), (0, 5), (0, -5)]
                    scheme_quality = scheme_speed + [(3, 0), (-3, 0), (0, 3), (0, -3)] + [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
                    for vy, vx in scheme_quality:
                        if vy == 0 and vx == 0:
                            continue

                        if yi + vy < 0 or yi + vy >= roi_h or xi + vx < 0 or xi + vx >= roi_w:
                            continue

                        M = img_mueller_base[yi + vy, xi + vx]
                        m = M.flatten()
                        pred = (pn @ m) / (pd @ m)
                        # loss = F.huber_loss(torch.tensor(gt), torch.tensor(pred), delta=delta).item()
                        loss = huber_loss(gt, pred, delta, 1 / np.abs(gt))

                        if loss < loss_best:
                            M_best = np.copy(M)
                            loss_best = float(loss)
                            is_updated = True

                    M = M_best * (0.01 * np.random.randn(4, 4) + 1)
                    M = M / M[0, 0]
                    M = GenericFilter(M)

                    # if np.random.rand() < 0.5:
                    #     M = np.eye(4)
                    #     dop = np.random.uniform(0.5, 1.0)
                    #     M[1, 1] = dop
                    #     M[2, 2] = dop
                    #     M[3, 3] = dop
                    #     M = M + np.random.randn(4, 4) * 0.01
                    #     M = np.clip(M, -1, 1)
                    #     M[0, 0] = 1.0

                    m = M.flatten()
                    pred = (pn @ m) / (pd @ m)
                    # loss = F.huber_loss(torch.tensor(gt), torch.tensor(pred), delta=delta).item()
                    loss = huber_loss(gt, pred, delta, 1 / np.abs(gt))
                    if loss < loss_best:
                        M_best = np.copy(M)
                        loss_best = float(loss)
                        is_updated = True

                    img_mueller_prop[yi, xi] = M_best

                    if is_updated:
                        num_updates += 1

                    loss_list.append(loss_best)

            pbar.set_postfix({"percentage": num_updates / (roi_h * roi_w) * 100, "loss": np.nanmedian(loss_list)})

            img_mueller_base = np.copy(img_mueller_prop)

            img_mueller_prop_vis = ee.mueller_image(img_mueller_prop)
            if img_mueller_prop_vis.shape[0] > 1080:
                img_mueller_prop_vis = cv2.resize(img_mueller_prop_vis, (512, 512))
            cv2.imshow("mueller_prop", img_mueller_prop_vis)
            key = cv2.waitKey(30)
            if key == ord("q"):
                break

        # # non-linear least squares
        # for yi in range(roi_h):
        #     for xi in range(roi_w):
        #         t_min = trig_t_x1[frame_i]
        #         t_max = trig_t_x1[frame_i + 1]
        #         t, p = events_fast.get(width // 2 + xi - roi_w // 2, height // 2 + yi - roi_h // 2, t_min, t_max)
        #         if len(t) <= 16:
        #             continue
        #         t = (t - t_min) / (t_max - t_min) * np.pi
        #         t_diff = np.convolve(t, [1, -1], mode="valid")
        #         # t_diff = t_diff - (t_refr / (t_max - t_min) * np.pi)
        #         t = np.convolve(t, [0.5, 0.5], mode="valid")
        #         p = (2 * p - 1)[1:]
        #         gt = 1 / t_diff * p * C

        #         pn, pd = ee.calcNumenatorDenominatorCoffs(t.astype(np.float32), phi1, phi2 - 5 * phi_offset)
        #         A = pn - np.diag(gt) @ pd

        #         m = img_mueller_DLT[yi, xi].flatten()

        #         def objective(m):
        #             pred = (pn @ m) / (pd @ m)
        #             return gt - pred

        #         try:
        #             res = least_squares(objective, m, method="lm", max_nfev=1000)
        #             M = res.x.reshape(4, 4)
        #         except ValueError:
        #             pass

        #         M_NLLS = M.copy()
        #         img_mueller_NLLS[yi, xi] = M_NLLS

        # # propagate for non-linear least squares
        # img_mueller_base = np.copy(img_mueller_NLLS)
        # img_mueller_prop_NLLS = np.copy(img_mueller_base)
        # for _ in trange(50, desc="Propagate NLLS"):
        #     for yi in range(roi_h):
        #         for xi in range(roi_w):
        #             t_min = trig_t_x1[frame_i]
        #             t_max = trig_t_x1[frame_i + 1]
        #             t, p = events_fast.get(width // 2 + xi - roi_w // 2, height // 2 + yi - roi_h // 2, t_min, t_max)
        #             if len(t) <= 16:
        #                 continue
        #             t = (t - t_min) / (t_max - t_min) * np.pi
        #             t_diff = np.convolve(t, [1, -1], mode="valid")
        #             # t_diff = t_diff - (t_refr / (t_max - t_min) * np.pi)
        #             t = np.convolve(t, [0.5, 0.5], mode="valid")
        #             p = (2 * p - 1)[1:]
        #             gt = 1 / t_diff * p * C

        #             pn, pd = ee.calcNumenatorDenominatorCoffs(t.astype(np.float32), phi1, phi2 - 5 * phi_offset)
        #             A = pn - np.diag(gt) @ pd

        #             M_best = img_mueller_base[yi, xi]
        #             m = M_best.flatten()
        #             pred = (pn @ m) / (pd @ m)
        #             loss_best = F.huber_loss(torch.tensor(gt), torch.tensor(pred)).item()

        #             scheme_speed = [(0, 1), (0, -1), (1, 0), (-1, 0), (5, 0), (-5, 0), (0, 5), (0, -5)]
        #             scheme_quality = scheme_speed + [(3, 0), (-3, 0), (0, 3), (0, -3)] + [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
        #             for vy, vx in scheme_quality:
        #                 if vy == 0 and vx == 0:
        #                     continue

        #                 if yi + vy < 0 or yi + vy >= roi_h or xi + vx < 0 or xi + vx >= roi_w:
        #                     continue

        #                 M = img_mueller_base[yi + vy, xi + vx]
        #                 m = M.flatten()
        #                 pred = (pn @ m) / (pd @ m)
        #                 loss = F.huber_loss(torch.tensor(gt), torch.tensor(pred)).item()

        #                 if loss < loss_best:
        #                     M_best = np.copy(M)
        #                     loss_best = float(loss)

        #             M = M_best * (1 + np.random.randn(4, 4) * 0.01)
        #             # M = M_best + np.random.randn(4, 4) * 0.01
        #             m = M.flatten()
        #             pred = (pn @ m) / (pd @ m)
        #             loss = F.huber_loss(torch.tensor(gt), torch.tensor(pred)).item()

        #             if loss < loss_best:
        #                 M_best = np.copy(M)

        #             img_mueller_prop_NLLS[yi, xi] = M_best

        #     img_mueller_base = np.copy(img_mueller_prop_NLLS)

        #     img_mueller_prop_NLLS_vis = ee.mueller_image(img_mueller_prop_NLLS)
        #     cv2.imshow("mueller_prop_NLLS", img_mueller_prop_NLLS_vis)
        #     key = cv2.waitKey(30)
        #     if key == ord("q"):
        #         break

        ee.imwrite_mueller(f"mueller_DLT_2-{frame_i}.png", img_mueller_DLT)
        ee.imwrite_mueller(f"mueller_NLLS_2-{frame_i}.png", img_mueller_NLLS)
        ee.imwrite_mueller(f"mueller_prop_2-{frame_i}.png", img_mueller_prop)
        # ee.imwrite_mueller(f"mueller_prop_NLLS_2-{frame_i}.png", img_mueller_prop_NLLS)

        img_M_opt_DLT_list.append(ee.mueller_image(img_mueller_DLT))
        img_M_opt_nonlinear_list.append(ee.mueller_image(img_mueller_NLLS))
        img_M_opt_DLT_prop_list.append(ee.mueller_image(img_mueller_prop))

        vmin = np.percentile(img_loss_DLT[img_loss_DLT < np.inf], 1)
        vmax = np.percentile(img_loss_DLT[img_loss_DLT < np.inf], 99)
        img_loss_DLT_vis = pa.applyColorMap(img_loss_DLT, "viridis", vmin, vmax)
        cv2.imwrite(f"loss_DLT-{frame_i}.png", img_loss_DLT_vis)

    cv2.imwrite("M_opt_DLT.png", pa.makeGrid(img_M_opt_DLT_list, border=8, border_color=(255, 255, 255)))
    cv2.imwrite("M_opt_nonlinear.png", pa.makeGrid(img_M_opt_nonlinear_list, border=8, border_color=(255, 255, 255)))
    cv2.imwrite("M_opt_DLT_prop.png", pa.makeGrid(img_M_opt_DLT_prop_list, border=8, border_color=(255, 255, 255)))

    # print("Non-linear least squares")
    # U, D, V = np.linalg.svd(A)
    # m = V[-1].reshape(4, 4).flatten()

    # w = np.ones(len(gt))
    # w = 1 / np.abs(gt)
    # for i in range(10):
    #     Aw = np.diag(w / np.sum(w)) @ A
    #     # Solve Ax=0 by svd
    #     U, D, V = np.linalg.svd(Aw)
    #     M = V[-1].reshape(4, 4)
    #     M = M / M[0, 0]
    #     M = np.clip(M, -1, 1)
    #     m = M.flatten()
    #     pred = (pn @ m) / (pd @ m)
    #     w = 1 / (np.abs(gt - pred) + 0.1)

    # m = np.diag([1, 0.8, 0.8, 0.8]).astype(np.float32).flatten()

    # m = pa.polarizer(0).astype(np.float32).flatten()

    # m = m / m[0]
    # m = np.clip(m, -0.9999, 0.9099)
    # m[0] = 1

    # print("Initial m")
    # print(np.reshape(m, (4, 4)))

    # def objective(m):
    #     pred = (pn @ m) / (pd @ m)
    #     return gt - pred

    # res = least_squares(objective, m, method="lm", verbose=1)
    # m = res.x

    # M = m.reshape(4, 4)
    # m = M.flatten()
    # print(M / M[0, 0])
    # pred = (pn @ m) / (pd @ m)
    # print(np.linalg.norm(gt - pred))
    # print(np.linalg.norm(A @ m))
    # print((A @ m).T @ A @ m)

    # img_M_opt_nonlinear = ee.mueller_image(M)
    # cv2.imwrite(f"M_opt_nonlinear-{frame_i}.png", img_M_opt_nonlinear)
    # img_M_opt_nonlinear_list.append(img_M_opt_nonlinear)

    exit()

    # plt.plot(t, gt, "o")
    # pred_DLT = ee.diffLn(M_DLT.flatten().astype(np.float32), t.astype(np.float32), phi1, phi2 - 5 * phi_offset)
    # plt.plot(t, pred_DLT, label="DLT")
    # pred = ee.diffLn(M.flatten().astype(np.float32), t.astype(np.float32), phi1, phi2 - 5 * phi_offset)
    # plt.plot(t, pred, label="Non-linear least squares")
    # plt.legend()
    # plt.savefig("calib_qwp_offset.png", bbox_inches="tight")
    # # plt.show()

    # exit()

    # Solve with calibration
    M = ee.fit(
        t.astype(np.float32),
        gt.astype(np.float32),
        phi1,
        phi2 - phi_offset,
        eps=1e-6,
        max_iter=10,
        debug=True,
    )
    print(np.reshape(M, (4, 4)))


if __name__ == "__main__":
    main()
