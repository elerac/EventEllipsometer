from pathlib import Path
import warnings
from typing import Optional, Dict, Any
import numpy as np
import numpy.typing as npt


def read_event(filepath: str, filepath_bias: Optional[str] = None) -> Dict[str, Any]:
    suffix = Path(filepath).suffix
    if suffix == ".raw":
        # Metavision raw format
        from metavision_core.event_io import EventsIterator

        mv_iterator = EventsIterator(filepath)
        height, width = mv_iterator.get_size()

        x_list = []
        y_list = []
        p_list = []
        t_list = []
        trig_p_list = []
        trig_t_list = []
        for evs in mv_iterator:
            evs = np.array(evs)
            if evs.size == 0:
                continue

            triggers = mv_iterator.reader.get_ext_trigger_events()
            if len(triggers) > 0:
                for trigger in triggers:
                    trig_p, trig_t, _ = trigger
                    trig_p_list.append(trig_p)
                    trig_t_list.append(trig_t)
                mv_iterator.reader.clear_ext_trigger_events()

            x = evs["x"]  # uint16
            y = evs["y"]  # uint16
            p = evs["p"]  # int16
            t = evs["t"]  # int64, [us]

            x_list.append(x)
            y_list.append(y)
            p_list.append(p)
            t_list.append(t)

        x = np.concatenate(x_list)
        y = np.concatenate(y_list)
        p = np.concatenate(p_list)
        t = np.concatenate(t_list)
        trig_p = np.array(trig_p_list)
        trig_t = np.array(trig_t_list)

        # Convert to dict
        data = {}
        data["x"] = x
        data["y"] = y
        data["p"] = p
        data["t"] = t
        data["trig_p"] = trig_p
        data["trig_t"] = trig_t
        data["width"] = width
        data["height"] = height

        # Bias values
        if filepath_bias is None:
            filepath_bias = Path(filepath).with_suffix(".bias")

        if Path(filepath_bias).exists():
            with open(filepath_bias, "r") as f:
                while True:
                    line = f.readline()
                    # Split by % and use it as key
                    if line == "":
                        break
                    value, key = line.split("%")
                    key = key.strip()
                    value = float(value)
                    data[key] = value

    elif suffix == ".npz":
        if filepath_bias is not None:
            warnings.warn("filepath_bias is ignored when reading .npz file")
        data_npz = np.load(filepath)
        data = {k: data_npz[k] for k in data_npz.files}  # Convert to dict
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    return data


def write_event(
    filepath: str,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    p: npt.ArrayLike,
    t: npt.ArrayLike,
    width: int = 1,
    height: int = 1,
    trigger: npt.ArrayLike = [],
    **metadata,
) -> None:
    suffix = Path(filepath).suffix

    # Add .npz if suffix is not provided
    if suffix == "":
        suffix = ".npz"
        filepath += suffix

    # Check if the suffix is supported
    if suffix != ".npz":
        raise ValueError(f"Unsupported file format: {suffix}")

    # Save as npzs
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    data = {"x": x, "y": y, "p": p, "t": t, "width": width, "height": height, "trigger": trigger}
    data.update(metadata)
    np.savez_compressed(filepath, **data)


def main():
    # raw2npz
    import argparse

    parser = argparse.ArgumentParser("Convert raw file to npz file")
    parser.add_argument("filepath_raw", type=str)
    args = parser.parse_args()
    filepath_raw = args.filepath_raw

    print("Converting raw file to npz file")
    print(f"Reading {filepath_raw}")
    events = read_event(filepath_raw)

    filepath_npz = Path(filepath_raw).with_suffix(".npz")

    print(f"Saved as {filepath_npz}")
    np.savez_compressed(filepath_npz, **events)


if __name__ == "__main__":
    main()
