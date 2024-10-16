# Calibration

## Contrast Threshold Calibration

`calib/thresh/on/*.npy` and `calib/thresh/off/*.npy` contain the data for the contrast threshold calibration.


```python
import numpy as np

bias_diff = 0
bias_diff_on = -20
bias_diff_off = -20
filename_C_on = f"calib/thresh/on/diff{bias_diff}_diff_on{bias_diff_on}_diff_off{bias_diff_off}.npy"
filename_C_off = f"calib/thresh/off/diff{bias_diff}_diff_on{bias_diff_on}_diff_off{bias_diff_off}.npy"
img_C_on = np.load(filename_C_on).astype(np.float32)
img_C_off = np.load(filename_C_off).astype(np.float32)
```

## QWP Offset Calibration

`calib/qwp_offset/` contains the data for the QWP offset calibration. See `phi1.txt` and `phi2.txt`.

```python
import numpy as np

filename_calib_phi1 = "calib/qwp_offset/phi1.txt"
filename_calib_phi2 = "calib/qwp_offset/phi2.txt"
phi1 = float(np.loadtxt(filename_calib_phi1, comments="%"))
phi2 = float(np.loadtxt(filename_calib_phi2, comments="%"))
```