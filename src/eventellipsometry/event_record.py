import argparse
import os
import datetime
from typing import Optional, Tuple, Union
from pathlib import Path
from metavision_core.event_io.raw_reader import initiate_device
from metavision_core.event_io import EventsIterator
from metavision_hal import I_ROI, DeviceConfig
import metavision_hal


def generate_filename_raw() -> Path:
    return Path(f"recording_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.raw")


def record(
    filepath: Optional[Union[str, Path]] = None,
    duration: float = 5.0,
    bias_diff: int = 0,
    bias_diff_on: int = 0,
    bias_diff_off: int = 0,
    bias_fo: int = 0,
    bias_hpf: int = 0,
    bias_refr: int = 0,
    roi: Optional[list[int]] = None,
    verbose: bool = True,
) -> Tuple[str, str]:
    """Record events and save to a file (.raw)

    Parameters
    ----------
    filepath : Optional[str, Path], optional
        Output file path, by default None. If None, the file will be saved in the "recordings" directory with the current timestamp.
    duration : float, optional
        Duration of recording [s], by default 5.0
    bias_diff : int, optional
        bias_diff, by default 0
    bias_diff_on : int, optional
        bias_diff_on, by default 0
    bias_diff_off : int, optional
        bias_diff_off, by default 0
    bias_fo : int, optional
        bias_fo, by default 0
    bias_hpf : int, optional
        bias_hpf, by default 0
    bias_refr : int, optional
        bias_refr, by default 0
    roi : Optional[list[int]], optional
        ROI: (x, y, width, height) or (width, height) or (width,), by default None
    verbose : bool, optional
        Print log messages, by default True

    Returns
    -------
    Tuple[str, str]
        File path of the recorded events (.raw) and bias values (.bias)
    """

    filepath_raw = filepath
    if filepath_raw is None:
        filepath_raw = Path("recordings") / generate_filename_raw()

    if Path(filepath_raw).suffix != ".raw":
        filepath_raw = Path(filepath_raw).with_suffix(".raw")

    if verbose:
        print(f"{datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')} Initializing device... ")
        print(f"  filepath={filepath}, duration={duration}, bias_diff={bias_diff}, bias_diff_on={bias_diff_on}, bias_diff_off={bias_diff_off}, bias_fo={bias_fo}, bias_hpf={bias_hpf}, bias_refr={bias_refr}, roi={roi} ")

    filepath_bias = Path(filepath_raw).with_suffix(".bias")

    if not Path(filepath_raw).parent.exists():
        if verbose:
            print(f"  Make directory: {Path(filepath_raw).parent}")
        Path(filepath_raw).parent.mkdir(parents=True)

    device_config = DeviceConfig()
    device_config.enable_biases_range_check_bypass(True)
    device = initiate_device("")

    # ROI
    # https://docs.prophesee.ai/stable/hw/manuals/region_of_interest.html
    if roi is not None:
        if len(roi) == 4:
            x, y, width, height = roi
        if len(roi) == 2:
            width, height = roi
            geometry = device.get_i_geometry()
            sensor_width, sensor_height = geometry.get_width(), geometry.get_height()
            x = (sensor_width - width) // 2
            y = (sensor_height - height) // 2
        else:
            raise ValueError("Invalid ROI format. ")
        device.get_i_roi().enable(True)
        device.get_i_roi().set_window(I_ROI.Window(x, y, width, height))

    # Enable trigger
    # https://docs.prophesee.ai/stable/hw/manuals/timing_interfaces.html
    i_trigger_in = device.get_i_trigger_in()
    i_trigger_in.enable(metavision_hal.I_TriggerIn.Channel.MAIN)

    # Biases
    # https://docs.prophesee.ai/stable/hw/manuals/biases.html
    device.get_i_ll_biases().set("bias_diff", bias_diff)
    device.get_i_ll_biases().set("bias_diff_on", bias_diff_on)  # the contrast threshold for ON events
    device.get_i_ll_biases().set("bias_diff_off", bias_diff_off)  # the contrast threshold for OFF events
    device.get_i_ll_biases().set("bias_fo", bias_fo)  # the low-pass filter
    device.get_i_ll_biases().set("bias_hpf", bias_hpf)  # the high-pass filter
    device.get_i_ll_biases().set("bias_refr", bias_refr)  # the refractory period

    device.get_i_events_stream().log_raw_data(str(filepath_raw))

    if verbose:
        print(f"{datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')} Recording events... ")

    max_duration_sec = duration
    mv_iterator = EventsIterator.from_device(device=device, max_duration=max_duration_sec * 1000000)
    # height, width = mv_iterator.get_size()

    # Statistics
    num_eve_pos = 0
    num_eve_neg = 0
    num_trig_pos = 0
    num_trig_neg = 0

    # Read events
    for evs in mv_iterator:
        # Events
        if evs.size != 0:
            p = evs["p"]
            num_eve_pos += len(p[p == 1])
            num_eve_neg += len(p[p == 0])

        # Triggers
        triggers = mv_iterator.reader.get_ext_trigger_events()
        if len(triggers) > 0:
            for trigger in triggers:
                trig_p, trig_t, _ = trigger
                if trig_p == 1:
                    num_trig_pos += 1

                if trig_p == 0:
                    num_trig_neg += 1

            mv_iterator.reader.clear_ext_trigger_events()

        if verbose:
            print(f"  Events: pos={num_eve_pos}, neg={num_eve_neg}, Triggers: pos={num_trig_pos}, neg={num_trig_neg}", end="\r")

    if verbose:
        print()

    device.get_i_events_stream().stop_log_raw_data()

    filesize = os.path.getsize(filepath_raw)

    if verbose:
        print(f"{datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')} Recording finished. ")
        print(f"  Exported to: {filepath_raw} ({filesize / 1024 / 1024:.2f} MB)")

    # Save bias values
    bias_diff = device.get_i_ll_biases().get("bias_diff")
    bias_diff_on = device.get_i_ll_biases().get("bias_diff_on")
    bias_diff_off = device.get_i_ll_biases().get("bias_diff_off")
    bias_fo = device.get_i_ll_biases().get("bias_fo")
    bias_hpf = device.get_i_ll_biases().get("bias_hpf")
    bias_refr = device.get_i_ll_biases().get("bias_refr")
    with open(filepath_bias, "w") as f:
        f.write(f"{bias_diff}    % bias_diff\n")
        f.write(f"{bias_diff_on}    % bias_diff_on\n")
        f.write(f"{bias_diff_off}    % bias_diff_off\n")
        f.write(f"{bias_fo}    % bias_fo\n")
        f.write(f"{bias_hpf}    % bias_hpf\n")
        f.write(f"{bias_refr}    % bias_refr\n")

    return str(filepath_raw), str(filepath_bias)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration of recording [s]")
    parser.add_argument("--bias_diff", "--diff", type=int, default=0, help="bias_diff")
    parser.add_argument("--bias_diff_on", "--diff_on", type=int, default=0, help="bias_diff_on")
    parser.add_argument("--bias_diff_off", "--diff_off", type=int, default=0, help="bias_diff_off")
    parser.add_argument("--bias_diff_on_off", "--diff_on_off", type=int, default=None, help="Set bias_diff_on and bias_diff_off to the same value")
    parser.add_argument("--bias_fo", "--fo", type=int, default=0, help="bias_fo")
    parser.add_argument("--bias_hpf", "--hpf", type=int, default=0, help="bias_hpf")
    parser.add_argument("--bias_refr", "--refr", type=int, default=0, help="bias_refr")
    parser.add_argument("--roi", type=int, nargs="+", default=None, help="ROI: (x, y, width, height) or (width, height) or (width,)")
    args = parser.parse_args()
    verbose = True

    if args.bias_diff_on_off is not None:
        args.bias_diff_on = args.bias_diff_on_off
        args.bias_diff_off = args.bias_diff_on_off

    record(args.output, args.duration, args.bias_diff, args.bias_diff_on, args.bias_diff_off, args.bias_fo, args.bias_hpf, args.bias_refr, args.roi, verbose)


if __name__ == "__main__":
    main()
