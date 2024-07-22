# EventEllipsometry

## Software Environment Setup

### Python Environment

This project is developed with rye for managing the python environment. To construct the python environment, run the `rye sync` command. 

### C++ Extension

This code has a CPP extension for the event data processing. To enable the extension, the extension needs to be built manually. The extension can be built by just running the `build_cpp.py` script. Or, run `build_cpp` command through rye. 
When building the extension, Eigen and OpenMP are required.

### Event Camera Operation

The event camera is operated by the Metavision SDK. The SDK can be installed by following the instruction in the [Metavision SDK](https://docs.prophesee.ai/stable/index.html). 


## Hardware Prototype Configuration

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

Assumption: 
- The motor has a encoder (two blades and one photointerrupter).
- Arduino measures the speed (frequency) of the motor by the encoder.
- The motor speed should be controlled by the Arduino with a PID controller.
- There are two motors: M1 and M5. The M5 motor rotates 5 times faster than M1.

Rules for triggering the camera:

- If the state is pos (by M1), then the M5 is allowed to produce a neg-edge.
- While the state is neg (by M5), then the M1 is allowed to produce a pos-edge.
