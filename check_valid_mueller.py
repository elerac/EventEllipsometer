import numpy as np
import numpy.typing as npt


def _movelastaxis(a: np.ndarray, source: int) -> np.ndarray:
    """Equivalent to `np.moveaxis(a, source, -1)` but does not move the axis if source is -1"""
    if source != -1:
        a = np.moveaxis(a, source, -1)
    return a


def isstokes(stokes: np.ndarray, atol: float = 1.0e-8, axis: int = -1) -> npt.NDArray[np.bool_]:
    """Check if the Stokes vector is physically valid.

    Parameters
    ----------
    stokes : np.ndarray
        Stokes vector.
    atol : float, optional
        Absolute tolerance, by default 1.0e-8.
    axis : int, optional
        The axis that contains the Stokes vectors, by default -1.

    Returns
    -------
    np.ndarray
        This is scalar if the input is a single Stokes vector, and an array of booleans if the input is a stack of Stokes vectors.

    Examples
    --------
    >>> s_a = np.array([1.0, 0.0, 0.0, 0.0])
    >>> s_b = np.array([1.0, 1.0, 0.0, 0.0])
    >>> s_c = np.array([1.0, 1.01, 0.0, 0.0])
    >>> s_abc = np.stack([s_a, s_b, s_c], axis=0)
    >>> isstokes(s_a)
    True
    >>> isstokes(s_b)
    True
    >>> isstokes(s_c)
    False
    >>> isstokes(s_abc)
    [ True  True False]
    """
    stokes = _movelastaxis(stokes, axis)
    s0 = stokes[..., 0]
    s1 = stokes[..., 1]
    s2 = stokes[..., 2]
    s3 = stokes[..., 3]

    # The intensity should be positive
    is_valid_intensity = s0 > 0

    # The DoP should be smaller than 1
    # The criterion is (s0**2 - (s1**2 + s2**2 + s3**2)) >= 0
    # but allow a small negative value due to numerical errors
    is_valid_dop = (s0**2 - (s1**2 + s2**2 + s3**2)) > -abs(atol)

    return np.bitwise_and(is_valid_intensity, is_valid_dop)


def ismueller(mueller: np.ndarray, method: str = "bf"):
    method = method.lower()
    if method == "bf":  # Brute-force
        for _ in range(10000):
            # Generate a random Stokes vector (dop = 1)
            s123 = np.random.uniform(-1, 1, 3)
            s123 = s123 / np.linalg.norm(s123)
            stokes_in = np.array([1.0, *s123])

            # Apply the Mueller matrix
            stokes_out = mueller @ stokes_in
            if not isstokes(stokes_out):
                import polanalyser as pa

                print("sin=", stokes_in, pa.cvtStokesToDoP(stokes_in))
                print("sout=", stokes_out, pa.cvtStokesToDoP(stokes_out))
                return False
        return True

    elif method == "gk":  # Givens-Kostinski, 1993
        G = np.diag([1.0, -1.0, -1.0, -1.0])
        M = mueller
        eigenvalues, eigenvectors = np.linalg.eigh(G @ M.T @ G @ M)

        # All eigenvalues should be real
        is_real = np.allclose(np.imag(eigenvalues), 0)

        if not is_real:
            return False

        # The eigenvectors S_{\sigma_1} corresponding to the largest eigenvalue should be a valid Stokes vector
        # Q1. How to determine the largest eigenvalue? Should I use abs()?
        # Q2. Should I normalize the Stokes vector?
        stokes_sigma1 = eigenvectors[:, np.argmax(np.abs(eigenvalues))]
        stokes_sigma1 = stokes_sigma1 / stokes_sigma1[0]
        is_stokes = isstokes(stokes_sigma1)

        return is_real and is_stokes

    else:  # https://github.com/usnistgov/pySCATMECH/blob/master/pySCATMECH/mueller.py
        M = mueller
        M = M + 1e-10 * M[0, 0] * np.diag([1, 0, 0, 0])
        G = np.diag([1.0, -1.0, -1.0, -1.0])
        MM = G @ M.T @ G @ M
        Q, W = np.linalg.eigh(MM)
        W = np.array(W.T)

        imax = 0
        max = -1e308
        ww = W[0]
        a = (Q @ np.conj(Q)) * 1e-14
        for q, w in zip(Q, W):
            if abs(q.imag) > a:
                return False
            if abs(q) > max:
                max = abs(q)
                ww = w.copy()

        ww = ww / ww[0]
        for i in range(1, 4):
            if abs(ww[i].imag) > a:
                return False
        if 1 < ww[1].real ** 2 + ww[2].real ** 2 + ww[3].real ** 2:
            return False
        return True


def main():
    np.set_printoptions(precision=3, suppress=True)

    while True:
        M = np.random.rand(4, 4) * 2 - 1
        M[0, 0] = 1.0
        # M = np.random.choice([-1, 1]) * M + 2 * np.linalg.norm(M, ord=2) * np.diag([1, 0, 0, 0])
        # M = M / M[0, 0]
        ismueller_bruteforce = ismueller(M, method="BF")
        ismueller_gk = ismueller(M, method="GK")
        print(f"valid={ismueller_bruteforce} (BF), valid={ismueller_gk} (GK)")
        if ismueller_bruteforce:
            break


if __name__ == "__main__":
    main()
