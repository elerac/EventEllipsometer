import numpy as np
from pathlib import Path

parent = f"{Path(__file__).parent}"


# Thresholds
def thresh(polarity, diff, diff_on, diff_off):
    filename_npy = f"{parent}/thresh/{polarity}/diff{diff}_diff_on{diff_on}_diff_off{diff_off}.npy"
    img_C = np.load(filename_npy)
    return img_C.astype(np.float32)


def thresh_on(diff, diff_on):
    return thresh("on", diff, diff_on, diff_on)


def thresh_off(diff, diff_off):
    return thresh("off", diff, diff_off, diff_off)


# QWP offset
filename_calib_phi1 = f"{parent}/qwp_offset/phi1.txt"
filename_calib_phi2 = f"{parent}/qwp_offset/phi2.txt"
phi1 = float(np.loadtxt(filename_calib_phi1, comments="%"))
phi2 = float(np.loadtxt(filename_calib_phi2, comments="%"))
