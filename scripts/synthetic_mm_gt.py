from pathlib import Path
import numpy as np
import polanalyser as pa
import cv2


def main():
    filepath_src = "rendered/bunny5"
    name = Path(filepath_src).name

    images, props = pa.imreadMultiple(filepath_src)
    mm_psg = props["mm_psg"]
    mm_psa = props["mm_psa"]
    mm_psg = np.array([pa.qwp(theta) @ pa.polarizer(0) for theta in props["theta"]])
    mm_psa = np.array([pa.polarizer(0) @ pa.qwp(5 * theta) for theta in props["theta"]])

    images = images[0::10]  # Downsample
    mm_psa = mm_psa[0::10]  # Downsample
    mm_psg = mm_psg[0::10]  # Downsample
    img_mueller = pa.calcMueller(images, mm_psg, mm_psa)
    print("Mueller matrix:", img_mueller.shape)

    img_mueller = img_mueller / img_mueller[..., 0, 0][..., None, None]
    img_mueller = img_mueller[:, :, 1, :, :]

    # Create mask (black background region is excluded)
    img_mueller_ = img_mueller.copy()
    img_mueller_[..., 0, 0] = 0
    img_mueller_[np.isnan(img_mueller_)] = 0
    img_mueller_[np.isinf(img_mueller_)] = 0
    img_mueller_sum = np.sum(np.abs(img_mueller_), axis=(-2, -1)).astype(np.float32)
    img_mueller_sum = cv2.medianBlur(img_mueller_sum, 3)
    img_mueller_mask = img_mueller_sum > 0.6
    img_mueller_mask = img_mueller_mask.astype(np.uint8) * 255
    cv2.imwrite(f"rendered/{name}_mm_mask.png", img_mueller_mask)

    # Fill the masked region with NaN
    img_mueller[img_mueller_mask == 0] = np.nan

    # Save the Mueller matrix
    np.save(f"rendered/{name}_mm_gt.npy", img_mueller)

    # Save visualization results
    img_mueller_vis = pa.applyColorMap(img_mueller, "RdBu", vmin=-1, vmax=1)
    img_mueller_vis[img_mueller_mask == 0] = 255
    img_mueller_vis_grid = pa.makeGridMueller(img_mueller_vis, border=0)
    cv2.imwrite(f"rendered/{name}_mm_gt.png", img_mueller_vis_grid)


if __name__ == "__main__":
    main()
