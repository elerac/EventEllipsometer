# EventEllipsometry

## Software Environment Setup

### Python Environment

(Not necessary) This project is developed with [rye](https://github.com/astral-sh/rye) for managing the python environment. To construct the python environment, run the `rye sync` command.

### C++ Extension

This code has a CPP extension for the event data processing. To enable the extension, the extension needs to be built manually. The extension can be built by just running the `build_cpp.py` script. Or, run `build_cpp` command through rye. 
When building the extension, Eigen and OpenMP are required.

### Event Camera Record

The event camera is operated by [Metavision SDK](https://docs.prophesee.ai/stable/index.html). Make sure to the SDK is installed and can be imported with Python (`import metavision_core`).

You can record the event data by running the `event_record.py` script, inside the `src/eventellipsometry` directory.

```bash
python src/eventellipsometry/event_record.py
```

### Synthetic Data Generation

#### Mitsuba3 

To render the synthetic data, [Mitsuba3](https://github.com/mitsuba-renderer/mitsuba3) renderer with '*_spectral_polarized' variant is required. This variant requires to build by yourself.

The scene files (3D model, pbsdf) need to be prepared in the `scenes` directory. 

- [bunny.ply](https://www.dropbox.com/scl/fi/jv8ncz9jrej1vc22ar9d2/20180310_KickAir8P_UVUnwrapped_Stanford_Bunny_OBJ-JPG.zip?rlkey=l8nczue360jrq3o5y79k9nzjh&e=1&dl=0)
  - As the original model does not have the UV mapping, the UV-mapped model is provided at [here](https://blenderartists.org/t/uv-unwrapped-stanford-bunny-happy-spring-equinox/1101297).
  - I convert this model to the PLY format by using Blender.
- [pbsdf](https://vclab.kaist.ac.kr/siggraph2020/pbrdfdataset/kaistdataset.html)

```bash
scenes
├── meshes
│   ├── bunny.ply
│   └── ...
├── pbsdf
│   ├── 9_blue_billard_mitsuba
│   │   ├── 9_blue_billard_inpainted.pbsdf
│   │   └── ...
│   └── ...
└── ...
```

The rendering can be done by running the `render_images.py` script.

```bash
python scripts/render_images.py
```

#### DVS Voltmeter

After render the images, the images are converted to the event data by the [DVS-Voltmeter](https://github.com/Lynn0306/DVS-Voltmeter) [Lin+, ECCV2022]. This repository self-contains the DVS-Voltmeter code that is modified for this project.

### Hardware Development

- [Arduino IDE](https://www.arduino.cc/en/software)
- [WaveForms (for Analog Discovery 3)](https://reference.digilentinc.com/reference/software/waveforms/waveforms-3/start)

## Hardware Prototype Configuration

### Event camera bias tuning

General guidelines for bias tuning:

- Refractory period should be small. (--bias_refr 235)
- High-frequency events should be recorded. (--bias_fo 55)

You can also refer to the following [officical page for bias tuning](https://docs.prophesee.ai/stable/hw/manuals/biases.html) for more information.

### Event camera and Arduino connection

Connection pin assignment:
| Pin Number | Wire Color | Signal Name | Signal Description |
|------------|------------|-------------|---------------------|
| 5 | Grey | TRIG_IN_P | Trigger input positive (Opto-coupler anode) |
| 6 | Pink | TRIG_IN_N | Trigger input negative (Opto-coupler cathode) |

- [Customizing your EVK4 cable for Triggering or Synchronization](https://support.prophesee.ai/portal/en/kb/articles/evk4-trigger-cable)
- [Metavision SDK Docs | Timing Interfaces](https://docs.prophesee.ai/stable/hw/manuals/timing_interfaces.html#trigger-interfaces)
- [Metavision Training Videos | Trigger and Synchronization Interfaces (YouTube)](https://youtu.be/EVoATjOM7co?si=G-gSWAB0OuK2Qz4z)

### Motors

#### Motor Setup and Control Assumptions

- The motor has a encoder (two blades and one photointerrupter).
- Arduino measures the speed (frequency) of the motor by the encoder.
- The motor speed should be controlled by the Arduino with a PID controller.
- There are two motors: M1 and M5. The M5 motor rotates 5 times faster than M1.

#### Rules of Trigger for Event Camera

The trigger is generated when Arduino detects the encoder signal (two times per rotation) of two motors. Arduino generates the trigger signal with the following rules:

- trigState is the state of the trigger signal. It can be either HIGH or LOW.
- If the state is HIGH and M5 encoder signal is detected, then the state is changed to LOW.
- If the state is LOW and M1 encoder signal is detected, then the state is changed to HIGH.
- Note that the Arduino's HIGH/LOW states are reversed in the event camera.
  - Arduino: digitalWrite(pin, HIGH); -> Event Camera: trig_p = 0
  - Arduino: digitalWrite(pin, LOW);  -> Event Camera: trig_p = 1


## MM evaluation
relative Frobenius distance

Aiello, Maximum-likelihood estimation of Mueller matrices, OPTICS LETTERS 2005.

$
\frac{||m - m^*||}{||m + m^*||}
$
