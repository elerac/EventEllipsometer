# Event Ellipsometer: Event-based Mueller-Matrix Video Imaging

[Ryota Maeda](https://elerac.github.io/), [Yunseong Moon](https://sites.google.com/view/yunseongmoon), [Seung-Hwan Baek](https://sites.google.com/view/shbaek/)

![teaser](docs/teaser_wide.png)

## Software Environment Setup

### Python Environment (Not Necessary)

This project was developed with [Rye](https://rye.astral.sh/) to manage the Python environment. To set up the Python environment, run the `rye sync` command.

### C++ Extension

We developed a C++ extension for the event data processing to accelerate the computation The extension needs to be built manually by running the `build_cpp.py`. When building the extension, Eigen and OpenMP are required.

### Event Camera and Data

[Metavision SDK](https://docs.prophesee.ai/stable/index.html) is necessary for acquiring data from event cameras and handling data I/O. Please ensure the SDK is installed and can be imported in Python using the command `import metavision_core`.

### Hardware Development

We developed a hardware prototype using [Arduino IDE](https://www.arduino.cc/en/software). To calibrate the contrast threshold, we utilized [WaveForms](https://reference.digilentinc.com/reference/software/waveforms/waveforms-3/start) to operate the Analog Discovery.
