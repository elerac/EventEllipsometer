import time
from typing import Optional
import numpy as np
import numpy.typing as npt
from tqdm import trange, tqdm


class EventMapPy:
    """Event data structure for fast access to t and p at (x, y) position."""

    def __init__(
        self,
        x: npt.NDArray[np.uint16],
        y: npt.NDArray[np.uint16],
        t: npt.NDArray[np.int64],
        p: npt.NDArray[np.int16],
        width: int,
        height: int,
        t_min: Optional[int] = None,
        t_max: Optional[int] = None,
        verbose: bool = False,
    ):
        start_time = time.time()
        if verbose:
            print(f"This is {self.__class__.__name__} class.")
            print("Converting event data...")

        # Check the size of the input data
        if len(x) != len(y) or len(x) != len(t) or len(x) != len(p):
            raise ValueError("x, y, t, p should have same size")

        # Check t is sorted
        is_sorted = np.all(np.diff(t) >= 0)
        if not is_sorted:
            raise ValueError("events must be sorted by time t.")

        # Slice the data if t_min and t_max are given
        if (t_min is not None) and (t_max is not None):
            index_min, index_max = np.searchsorted(t, [t_min, t_max], side="right")
            x = x[index_min:index_max]
            y = y[index_min:index_max]
            t = t[index_min:index_max]
            p = p[index_min:index_max]
        elif (t_min is not None) or (t_max is not None):
            raise ValueError("Both t_min and t_max should be given.")

        # Append the (t, p) to the corresponding spatial coordinates
        num = len(x)
        self.map_t = [[[] for _ in range(width)] for _ in range(height)]
        self.map_p = [[[] for _ in range(width)] for _ in range(height)]
        for xi, yi, ti, pi in tqdm(zip(x, y, t, p), total=num, disable=not verbose, desc="Gather t and p (xy)"):
            self.map_t[yi][xi].append(ti)
            self.map_p[yi][xi].append(pi)

        # Convert list to ndarray
        dtype_t = t.dtype
        dtype_p = p.dtype
        for j in trange(height, disable=not verbose, desc=f"Convert to ndarray ({width}x{height})"):
            for i in range(width):
                self.map_t[j][i] = np.array(self.map_t[j][i], dtype=dtype_t)
                self.map_p[j][i] = np.array(self.map_p[j][i], dtype=dtype_p)

        if verbose:
            print("Data conversion completed.")
            elapsed_time = time.time() - start_time
            print(f"Elapsed time: {elapsed_time:.2f}s")

    def get(self, ix: int, iy: int, t_min: Optional[int] = None, t_max: Optional[int] = None) -> tuple[npt.NDArray, npt.NDArray]:
        t_xy = self.map_t[iy][ix]
        p_xy = self.map_p[iy][ix]

        if len(t_xy) == 0:
            return np.array([]), np.array([])

        if not (t_min is None and t_max is None):
            t_min = t_min if t_min is not None else np.min(t_xy)
            t_max = t_max if t_max is not None else np.max(t_xy)
            index_min, index_max = np.searchsorted(t_xy, [t_min, t_max], side="right")
            t_xy = t_xy[index_min:index_max]
            p_xy = p_xy[index_min:index_max]

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
    t = np.random.randint(0, 100000, num) * 1000
    p = np.random.randint(0, 2, size=num)

    print("-" * 20)
    print("x:", x.dtype, x.shape)
    print("y:", y.dtype, y.shape)
    print("t:", t.dtype, t.shape)
    print("p:", p.dtype, p.shape)
    print("width:", width)
    print("height:", height)
    print("-" * 20)

    eventmap = EventMapPy(x, y, t, p, width, height, verbose=True)

    print("-" * 20)
    print("Test get method")
    for _ in range(10):
        ix, iy = np.random.randint(0, width), np.random.randint(0, height)
        t_ixy, p_ixy = eventmap.get(ix, iy)

        print(f"({ix:4d}, {iy:4d}) - Num of events: {len(t_ixy)}, t: {t_ixy}")


if __name__ == "__main__":
    main()
