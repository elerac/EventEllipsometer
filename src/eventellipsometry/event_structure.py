import time
from typing import Optional, Dict, Any
import numpy as np
import numpy.typing as npt
from tqdm import trange, tqdm

try:
    from ._eventellipsometry_impl import cvtEventStrucure

    cpp_available = True
except ImportError:
    cpp_available = False


class FastEventAccess:
    """Event data structure for fast access to t and p at (x, y) position."""

    def __init__(
        self,
        events: Dict[str, Any],
        t_min: Optional[int] = None,
        t_max: Optional[int] = None,
        verbose: bool = False,
    ):
        start_time = time.time()
        if verbose:
            print(f"This is {self.__class__.__name__} class.")
            print("Converting event data...")

        x = events["x"]
        y = events["y"]
        t = events["t"]
        p = events["p"]
        width = events["width"]
        height = events["height"]

        if not (t_min is None and t_max is None):
            t_min = t_min if t_min is not None else np.min(t)
            t_max = t_max if t_max is not None else np.max(t)
            index = np.where((t_min <= t) & (t < t_max))
            x = x[index]
            y = y[index]
            t = t[index]
            p = p[index]

        # Sort by t
        sort_idx = np.argsort(t)
        x = x[sort_idx]
        y = y[sort_idx]
        t = t[sort_idx]
        p = p[sort_idx]

        # Append t and p to each (x, y) position
        if cpp_available:
            self.t_img, self.p_img = cvtEventStrucure(x, y, t, p, width, height, t_min, t_max)
        else:
            num = len(x)
            self.t_img = [[[] for _ in range(width)] for _ in range(height)]
            self.p_img = [[[] for _ in range(width)] for _ in range(height)]
            for xi, yi, ti, pi in tqdm(zip(x, y, t, p), total=num, disable=not verbose, desc="Gather t and p (xy)"):
                self.t_img[yi][xi].append(ti)
                self.p_img[yi][xi].append(pi)

        # Convert to ndarray
        dtype_t = t.dtype
        dtype_p = p.dtype
        for j in trange(height, disable=not verbose, desc=f"Convert to ndarray ({width}x{height})"):
            for i in range(width):
                self.t_img[j][i] = np.array(self.t_img[j][i], dtype=dtype_t)
                self.p_img[j][i] = np.array(self.p_img[j][i], dtype=dtype_p)

        if verbose:
            print("Data conversion completed.")
            elapsed_time = time.time() - start_time
            print(f"Elapsed time: {elapsed_time:.2f}s")

    def get(self, ix: int, iy: int, t_min: Optional[int] = None, t_max: Optional[int] = None) -> tuple[npt.NDArray, npt.NDArray]:
        t_xy = self.t_img[iy][ix]
        p_xy = self.p_img[iy][ix]

        if len(t_xy) == 0:
            return np.array([]), np.array([])

        if not (t_min is None and t_max is None):
            t_min = t_min if t_min is not None else np.min(t_xy)
            t_max = t_max if t_max is not None else np.max(t_xy)
            left, right = np.searchsorted(t_xy, [t_min, t_max])
            t_xy = t_xy[left:right]
            p_xy = p_xy[left:right]

        return t_xy, p_xy


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--num", type=int, default=2000000, help="Number of events")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    np.set_printoptions(precision=1, suppress=True)
    num = args.num
    width, height = 1280, 720
    x = np.random.randint(0, width, size=num).astype(np.uint16)
    y = np.random.randint(0, height, size=num).astype(np.uint16)
    t = np.random.rand(num) * 1000
    p = np.random.randint(0, 2, size=num)

    print("-" * 20)
    print("x:", x.dtype, x.shape)
    print("y:", y.dtype, y.shape)
    print("t:", t.dtype, t.shape)
    print("p:", p.dtype, p.shape)
    print("width:", width)
    print("height:", height)
    print("-" * 20)

    events = {"x": x, "y": y, "t": t, "p": p, "width": width, "height": height}

    events_fast = FastEventAccess(events, verbose=True)

    print("-" * 20)
    print("Test get method")
    for _ in range(10):
        ix, iy = np.random.randint(0, width), np.random.randint(0, height)
        t_ixy, p_ixy = events_fast.get(ix, iy)

        print(f"({ix:4d}, {iy:4d}) - Num of events: {len(t_ixy)}, t: {t_ixy}")


if __name__ == "__main__":
    main()
