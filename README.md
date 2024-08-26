# EventEllipsometry

## Software Environment Setup

### Python Environment

This project is developed with [rye](https://github.com/astral-sh/rye) for managing the python environment. To construct the python environment, run the `rye sync` command. 

### C++ Extension

This code has a CPP extension for the event data processing. To enable the extension, the extension needs to be built manually. The extension can be built by just running the `build_cpp.py` script. Or, run `build_cpp` command through rye. 
When building the extension, Eigen and OpenMP are required.

### Event Camera Operation

The event camera is operated by the Metavision SDK. The SDK can be installed by following the instruction in the [Metavision SDK](https://docs.prophesee.ai/stable/index.html).

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
