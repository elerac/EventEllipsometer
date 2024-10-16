# Calibration

## Contrast Threshold Calibration

`calib/thresh/on/*.npy` and `calib/thresh/off/*.npy` contain the data for the contrast threshold calibration.


```python
import calib

img_Con = calib.thresh_on(0, -20)
img_Coff = calib.thresh_off(0, -20)
```

## QWP Offset Calibration

`calib/qwp_offset/` contains the data for the QWP offset calibration. See `phi1.txt` and `phi2.txt`.

```python
import calib

phi1 = calib.phi1
phi2 = calib.phi2
```