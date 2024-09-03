import argparse
import os
import datetime
import time
import warnings
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
    duration: Optional[float] = 5.0,
    bias_diff: int = 0,
    bias_diff_on: int = 0,
    bias_diff_off: int = 0,
    bias_fo: int = 0,
    bias_hpf: int = 0,
    bias_refr: int = 0,
    roi: Optional[list[int]] = None,
    delta_t: float = 0.05,
    verbose: bool = True,
) -> Tuple[str, str]:
    """Record events and save to a file (.raw)

    Parameters
    ----------
    filepath : Optional[str, Path], optional
        Output file path, by default None. If None, the file will be saved in the "recordings" directory with the current timestamp.
    duration : float, optional
        Duration of recording [s], by default 5.0. If None, the recording will continue until the user stops it.
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
    delta_t : float, optional
        Time interval [s] to read events, by default 0.05
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
        print(f"  filepath={filepath}, duration={duration}, bias_diff={bias_diff}, bias_diff_on={bias_diff_on}, bias_diff_off={bias_diff_off}, bias_fo={bias_fo}, bias_hpf={bias_hpf}, bias_refr={bias_refr}, roi={roi}, delta_t={delta_t}")

    filepath_bias = Path(filepath_raw).with_suffix(".bias")

    if not Path(filepath_raw).parent.exists():
        if verbose:
            print(f"  Make directory: {Path(filepath_raw).parent}")
        Path(filepath_raw).parent.mkdir(parents=True)

    device_config = DeviceConfig()
    device_config.enable_biases_range_check_bypass(True)

    # Connect to the camera
    max_retries = 5
    for i in range(max_retries):
        try:
            device = initiate_device("")
        except OSError as e:
            if i == (max_retries - 1):
                raise e
            warnings.warn(f"Failed to initialize device. Retry {i + 1}/{max_retries}.", stacklevel=2)
            time.sleep(1)
            continue
        break

    # ROI
    # https://docs.prophesee.ai/stable/hw/manuals/region_of_interest.html
    if roi is not None:
        len_roi = 0
        if hasattr(roi, "__len__"):
            len_roi = len(roi)
        elif isinstance(roi, int):
            len_roi = 1
            roi = [roi]

        if len_roi not in [1, 2, 4]:
            raise ValueError(f"Invalid ROI: {roi}")

        if len_roi == 4:
            x, y, width, height = roi

        if len_roi == 1:
            roi = [roi[0], roi[0]]
            len_roi = len(roi)

        if len_roi == 2:
            width, height = roi
            geometry = device.get_i_geometry()
            sensor_width, sensor_height = geometry.get_width(), geometry.get_height()
            x = (sensor_width - width) // 2
            y = (sensor_height - height) // 2

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
    if max_duration_sec is not None:
        max_duration = max_duration_sec * 1000000  # [s] -> [us]
    else:
        max_duration = None
    mv_iterator = EventsIterator.from_device(device=device, max_duration=max_duration, delta_t=int(delta_t * 1000000))
    # height, width = mv_iterator.get_size()

    # Statistics
    num_evs_pos = 0
    num_evs_neg = 0
    total_evs_pos = 0
    total_evs_neg = 0
    total_trig_pos = 0
    total_trig_neg = 0

    # Read events
    is_printed = False
    for evs in mv_iterator:
        # Events
        if evs.size != 0:
            p = evs["p"]
            num_evs_pos = len(p[p == 1])
            num_evs_neg = len(p[p == 0])
            total_evs_pos += num_evs_pos
            total_evs_neg += num_evs_neg

        # Triggers
        triggers = mv_iterator.reader.get_ext_trigger_events()
        if len(triggers) > 0:
            for trigger in triggers:
                trig_p, trig_t, _ = trigger
                if trig_p == 1:
                    total_trig_pos += 1

                if trig_p == 0:
                    total_trig_neg += 1

            mv_iterator.reader.clear_ext_trigger_events()

        if verbose:
            if is_printed:
                # Move cursor up
                print("\033[F" * 4, end="")

            columns = os.get_terminal_size().columns

            # Statistics (filled with spaces at the left side)
            total_evs = total_evs_pos + total_evs_neg
            if total_evs > 0:
                percent_total_pos = total_evs_pos / total_evs * 100
                percent_total_neg = total_evs_neg / total_evs * 100
            else:
                percent_total_pos = "N/A"
                percent_total_neg = "N/A"
            num_evs = num_evs_pos + num_evs_neg
            if num_evs > 0:
                percent_num_pos = num_evs_pos / num_evs * 100
                percent_num_neg = num_evs_neg / num_evs * 100
            else:
                percent_num_pos = "N/A"
                percent_num_neg = "N/A"
            print(f"  Events(total): pos={total_evs_pos} ({percent_total_pos:.1f}%), neg={total_evs_neg} ({percent_total_neg:.1f}%)".ljust(columns))
            print(f"  Events({delta_t:.2f}s): pos={num_evs_pos} ({percent_num_pos:.1f}%), neg={num_evs_neg} ({percent_num_neg:.1f}%)".ljust(columns))
            print(f"  Triggers: pos={total_trig_pos}, neg={total_trig_neg}".ljust(columns))

            if num_evs_neg > total_evs_neg:
                print("  Warning: num_evs_neg > total_evs_neg")
                exit()

            # Progress bar
            current_time = int(mv_iterator.get_current_time()) / 1000000
            if max_duration_sec is not None:
                block = "█"
                percent = min(current_time / max_duration_sec, 1.0)
                msg0 = f"  {current_time:.1f}/{max_duration_sec:.1f}s |"
                msg1 = f"|{percent * 100:.1f}%  "
                columns_bar = columns - len(msg0) - len(msg1) - 1
                num_blocks = int(percent * columns_bar)
                num_empty = columns_bar - num_blocks
                print(f"{msg0}{block * num_blocks}{' ' * num_empty}{msg1}")
            else:
                print(f"  Current time: {current_time:.1f}s")

            is_printed = True

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
    parser.add_argument("--delta_t", type=float, default=0.05, help="Time interval [s] to read events")
    parser.add_argument("--roi", type=int, nargs="+", default=None, help="ROI: (x, y, width, height) or (width, height) or (width,)")
    args = parser.parse_args()
    verbose = True

    if args.bias_diff_on_off is not None:
        args.bias_diff_on = args.bias_diff_on_off
        args.bias_diff_off = args.bias_diff_on_off

    if args.duration <= 0:
        args.duration = None

    record(args.output, args.duration, args.bias_diff, args.bias_diff_on, args.bias_diff_off, args.bias_fo, args.bias_hpf, args.bias_refr, args.roi, args.delta_t, verbose)


if __name__ == "__main__":
    main()
