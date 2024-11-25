from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

import polanalyser as pa

from DVS_Voltmeter import EventSim
from DVS_Voltmeter import camera_para

import eventellipsometer as ee


def main():
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set camera parameters
    camera_type = "DVS346"
    # camera_type = "Clean"
    k1, k2, k3, k4, k5, k6 = camera_para[camera_type]
    k1 = k1 * 8  # Increase the sensitivity
    params = [k1, k2, k3, k4, k5, k6]
    print(f"Parameters: {params}")
    thresh = 1 / k1
    sim = EventSim(*params)

    # Read images
    filepath_src = Path("rendered/bunny5")
    print("Reading images from", filepath_src)
    images, props = pa.imreadMultiple(filepath_src)
    print("Input images:", images.shape)
    print("Properties:", props.keys())
    N, H, W, C = images.shape

    times = np.linspace(0, 1 / 30, N) * 1e6  # [us]

    events = []
    trig_t = []
    trig_p = []
    pbar = tqdm(times)
    num_events, num_on_events, num_off_events = 0, 0, 0
    for i, t in enumerate(pbar):
        if i == 0 or i == N - 1:
            trig_t.append(t)
            trig_p.append(0)
            trig_t.append(t)
            trig_p.append(1)

        img_bgr = images[i]
        img_mono = img_bgr[:, :, 1]
        img_mono = img_mono * 255 * 10000  # brighten the image

        event = sim.generate_events(img_mono, t)
        if event is None:
            continue

        events.append(event)

        num_events += event.shape[0]
        num_on_events += np.sum(event[:, -1] == 1)
        num_off_events += np.sum(event[:, -1] == 0)
        pbar.set_description(f"{num_events} = ON: {num_on_events} + OFF: {num_off_events}")

    events = np.concatenate(events, axis=0)
    t = events[:, 0].astype(np.int64)  # / 1e6  # [s]
    x = events[:, 1].astype(np.uint16)
    y = events[:, 2].astype(np.uint16)
    p = events[:, 3].astype(np.int16)
    trig_t = np.array(trig_t, dtype=np.int64)
    trig_p = np.array(trig_p, dtype=np.int16)

    filepath_dst = filepath_src.parent / (filepath_src.name + ".npz")
    ee.write_event(filepath_dst, x, y, p, t, width=W, height=H, trig_t=trig_t, trig_p=trig_p, mode="synthetic", thresh=thresh, camera_para=params)
    print("Events saved to", filepath_dst)


if __name__ == "__main__":
    main()
