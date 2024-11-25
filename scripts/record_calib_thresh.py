import itertools
import datetime
from tqdm import trange, tqdm
import eventellipsometer as ee


def main():
    start = datetime.datetime.now()
    print(f"Start recording at {start}")
    duration = 17
    bias_diff_list = [20]  # List of bias_diff
    bias_diff_on_off = [-35]  # List of bias_diff_on and bias_diff_off
    bias_fo = 55
    bias_hpf = 0
    bias_refr = 235
    width = 1280
    height = 720
    w = 80
    h = 80
    name = "calib_thresh_rampdown"

    product_list = list(x for x in itertools.product(bias_diff_list, bias_diff_on_off))
    print(product_list)

    pbar = tqdm(product_list, disable=True)
    for bias_diff, bias_diff_on_off in pbar:
        bias_diff_on = bias_diff_on_off
        bias_diff_off = bias_diff_on_off
        dirname = f"duration{duration}_diff{bias_diff}_diff_on{bias_diff_on}_diff_off{bias_diff_off}_fo{bias_fo}_hpf{bias_hpf}_refr{bias_refr}"
        pbar.set_description(f"Recording {dirname}")
        for y in trange(0, height, h):
            for x in trange(0, width, w, leave=False):

                roi = (x, y, w, h)
                filepath = f"recordings/{name}/{dirname}/x{x}_y{y}_w{w}_h{h}.raw"

                kwargs = {
                    "filepath": filepath,
                    "duration": duration,
                    "bias_diff": bias_diff,
                    "bias_diff_on": bias_diff_on,
                    "bias_diff_off": bias_diff_off,
                    "bias_fo": bias_fo,
                    "bias_hpf": bias_hpf,
                    "bias_refr": bias_refr,
                    "roi": roi,
                    "verbose": True,
                }
                ee.record(**kwargs)

    print(f"Start recording at {start}")
    print(f"End recording at {datetime.datetime.now()}")


if __name__ == "__main__":
    main()
