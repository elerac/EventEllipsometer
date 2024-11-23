import numpy as np
import cv2
from pathlib import Path
from tqdm import trange


def main():
    filename_npz = "rendered/bunny5.npz"
    filename_mask = "rendered/bunny5_mm_mask.png"

    event = np.load(filename_npz)
    img_mask = cv2.imread(filename_mask, cv2.IMREAD_GRAYSCALE)

    cv2.imshow("Mask", img_mask)
    cv2.waitKey(30)

    event = dict(event)
    x = event["x"]
    y = event["y"]
    p = event["p"]
    t = event["t"]

    index = np.ones(len(x), dtype=bool)
    for i in trange(len(x)):
        if img_mask[y[i], x[i]] < 128:
            index[i] = False

    event["x"] = x[index]
    event["y"] = y[index]
    event["p"] = p[index]
    event["t"] = t[index]

    filename_dst = Path(filename_npz).parent / (Path(filename_npz).stem + "_masked.npz")
    np.savez_compressed(filename_dst, **event)


if __name__ == "__main__":
    main()
