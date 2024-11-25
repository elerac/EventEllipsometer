import numpy as np
import eventellipsometer as ee


def add_noise(ellipsometry_eventmaps: list[ee.EventEllipsometryDataFrame]) -> list[ee.EventEllipsometryDataFrame]:
    for ellipsometry_eventmap in ellipsometry_eventmaps:
        height = ellipsometry_eventmap.shape(0)
        width = ellipsometry_eventmap.shape(1)
        for j in range(height):
            for i in range(width):
                theta, dlogI, weights, phi_offset = ellipsometry_eventmap.get(i, j)
                # Small noise
                dlogI = dlogI + np.random.normal(0, 0.5, dlogI.shape)
                # Outlier noise
                ratio = 0.03
                outlier = np.random.choice([True, False], size=dlogI.shape, p=[ratio, 1 - ratio])
                dlogI[outlier] = np.random.normal(0, 5, dlogI[outlier].shape)
                ellipsometry_eventmap.set(i, j, theta, dlogI, weights, phi_offset)
    return ellipsometry_eventmaps
