import numpy as np
import polanalyser as pa
import time

ISMUELLER_STOKES = "ISMUELLER_STOKES"  # Stokes criterion by brute-force
ISMUELLER_GK = "ISMUELLER_GK"  # Givens-Kostinski, 1993


def ismueller(mueller: np.ndarray, method: str = ISMUELLER_STOKES):
    if method == ISMUELLER_STOKES:
        """Check stokes criterion for Mueller matrix with brute-force approach."""
        num = 100000
        chunk_num = 10
        count = 0
        mask = np.full(mueller.shape[:-2], True, dtype=bool)
        while True:
            # Project random Stokes vectors to the Mueller matrix
            # the output should be Stokes vectors
            s_in = pa.random.stokes(dop=1.0, size=chunk_num)  # (num, 4)
            s_out = np.einsum("...ij,...kj->...ki", mueller[mask], s_in, optimize="optimal")  # (..., num, 4)
            isstokes = pa.isstokes(s_out)  # (..., num)
            mask[mask] = np.all(isstokes, axis=-1)  # (...)

            # Check the exit condition
            count += chunk_num
            if count >= num:
                break

            # Increase the chunk size
            chunk_num = min(chunk_num * 2, num - count)

        return mask

    elif method == ISMUELLER_GK:  # Givens-Kostinski, 1993
        G = np.diag([1.0, -1.0, -1.0, -1.0])  # (4, 4)
        M = mueller  # (..., 4, 4)
        M_T = np.moveaxis(M, -1, -2)  # (..., 4, 4)
        eigenvalues, eigenvectors = np.linalg.eigh(G @ M_T @ G @ M)

        # All eigenvalues should be real
        is_real = np.allclose(np.imag(eigenvalues), 0)

        # The eigenvectors S_{\sigma_1} corresponding to the largest eigenvalue should be a valid Stokes vector
        # Q1. How to determine the largest eigenvalue? Should I use abs()?
        # Q2. Should I normalize the Stokes vector?
        index = np.argmax(np.abs(eigenvalues), axis=-1)  # (...,)
        stokes_sigma1 = np.take_along_axis(eigenvectors, index[..., None, None], axis=-2)[..., 0, :]  # (..., 4)
        stokes_sigma1 = stokes_sigma1 / stokes_sigma1[..., 0][..., None]
        is_stokes = pa.isstokes(stokes_sigma1)  # (...,)
        return np.bitwise_and(is_real, is_stokes)

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
    np.random.seed(0)

    M = np.random.rand(1000, 100, 4, 4) * 2 - 1
    M[..., 0, 0] = 1.0

    time_start = time.time()
    ismueller_bf = ismueller(M, method=ISMUELLER_STOKES)
    time_end = time.time()
    print(f"Time (BF): {time_end - time_start:.3f} sec")

    time_start = time.time()
    ismueller_gk = ismueller(M, method=ISMUELLER_GK)
    time_end = time.time()
    print(f"Time (GK): {time_end - time_start:.3f} sec")

    print(f"valid={np.sum(ismueller_bf)} (BF), valid={np.sum(ismueller_gk)} (GK)")


if __name__ == "__main__":
    main()
