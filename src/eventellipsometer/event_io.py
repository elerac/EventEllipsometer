from pathlib import Path
import warnings
from typing import Optional, Dict, Any, Union
import numpy as np
import numpy.typing as npt


def read_event(filepath: Union[str, Path], filepath_bias: Optional[str] = None) -> Dict[str, Any]:
    suffix = Path(filepath).suffix
    filepath = str(filepath)
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
        to_scalar_if_possible = lambda x: x.item() if np.ndim(x) == 0 else x
        data = {k: to_scalar_if_possible(data_npz[k]) for k in data_npz.files}  # Convert to dict
    else:
        raise ValueError(f"Unsupported file format: '{suffix}'")

    return data


def write_event(filename_npz: Union[str, Path], x: npt.ArrayLike, y: npt.ArrayLike, p: npt.ArrayLike, t: npt.ArrayLike, **metadata):
    """Save event data as npz file

    Parameters
    ----------
    filename_npz : Union[str, Path]
        Filename to save as npz file
    x : npt.ArrayLike
        x-coordinate of events
    y : npt.ArrayLike
        y-coordinate of events
    p : npt.ArrayLike
        Polarity of events
    t : npt.ArrayLike
        Timestamp of events
    metadata : Dict[str, Any]
        Metadata to save as npz file

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 3], dtype=np.uint16)
    >>> y = np.array([4, 5, 6], dtype=np.uint16)
    >>> p = np.array([1, 0, 1], dtype=np.int16)
    >>> t = np.array([100, 200, 300], dtype=np.int64)
    >>> write_event("test.npz", x, y, p, t, width=640, height=480)
    >>> events = np.load("test.npz")
    >>> print(events.files)
    ['x', 'y', 'p', 't', 'width', 'height']
    """
    filename_npz = Path(filename_npz)
    suffix = filename_npz.suffix

    # Add .npz if suffix is not provided
    if suffix != ".npz":
        filename_npz.with_suffix(".npz")

    # Save as npzs
    filename_npz.parent.mkdir(parents=True, exist_ok=True)
    data = {"x": x, "y": y, "p": p, "t": t}
    data.update(metadata)
    np.savez_compressed(filename_npz, **data)


def main():
    # raw2npz
    import argparse
    import glob

    parser = argparse.ArgumentParser("Convert raw file(s) to npz file(s)")
    parser.add_argument("filepath_raw", type=str, help="Path to raw file or regex pattern")
    args = parser.parse_args()
    filepath_raw = args.filepath_raw
    print(f"filepaths_raw: {filepath_raw}")

    is_regex = "*" in filepath_raw
    if is_regex:
        filepath_raw = glob.glob(filepath_raw)
    else:
        filepath_raw = [filepath_raw]

    for i, filepath in enumerate(filepath_raw):
        print(f"[{i}/{len(filepath_raw)}]")
        print(f"  src : {filepath}")
        events = read_event(filepath)

        filepath_npz = Path(filepath).with_suffix(".npz")

        print(f"  dst: {filepath_npz}")
        np.savez_compressed(filepath_npz, **events)


if __name__ == "__main__":
    main()
