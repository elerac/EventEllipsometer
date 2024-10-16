import cv2
import numpy as np
import polanalyser as pa
import matplotlib.pyplot as plt
import calib


def main():
    np.set_printoptions(precision=3, suppress=True)
    diff = [0, 10, 15, 20]
    diff_on = [0, -10, -20, -35]
    diff_off = [0, -10, -20, -35]

    # Load images
    print("Loading images...")
    dict_img_Con = {}
    dict_img_Coff = {}
    for d in diff:
        for do, dof in zip(diff_on, diff_off):
            img_Con = calib.thresh_on(d, do)
            img_Coff = calib.thresh_off(d, dof)

            img_Con = img_Con
            img_Coff = img_Coff

            dict_img_Con[(d, do)] = img_Con
            dict_img_Coff[(d, dof)] = img_Coff

    # Plot hist
    print("Plotting histograms...")
    for d in diff:
        plt.clf()
        bins = 500
        alpha = 0.5
        colors = plt.get_cmap("tab10").colors
        for i, do in enumerate(diff_on):
            img_Con = dict_img_Con[(d, do)]
            color = colors[i]

            hist, binedges = np.histogram(img_Con.flatten(), bins=bins)
            bincenters = 0.5 * (binedges[1:] + binedges[:-1])
            plt.plot(bincenters, hist, color=color, label=f"diff{d}_on{do}")

            # plt.hist(img_Con.flatten(), bins=bins, alpha=alpha, color=color, label=f"diff{d}_on{do}")

        for i, dof in enumerate(diff_off):
            img_Coff = dict_img_Coff[(d, dof)]

            print(f"diff{d}_off{dof}:")
            print(img_Coff[:10, :10])
            print(np.var(img_Coff))
            center = np.median(img_Coff)
            color = colors[i]
            hist, binedges = np.histogram(img_Coff.flatten(), bins=bins)
            bincenters = 0.5 * (binedges[1:] + binedges[:-1])
            plt.plot(bincenters, hist, color=color, label=f"diff{d}_off{dof}")
            # plt.hist(img_Coff.flatten() - center, bins=bins, alpha=alpha, color=color, label=f"diff{d}_off{dof}")

        plt.xlim(-0.5, 0.5)
        plt.legend()
        plt.savefig(f"fig_hist_diff{d}.png", bbox_inches="tight", dpi=500)

    # Export visualized images
    print("Exporting visualized images...")
    vmin = -0.5
    vmax = 0.5
    for (d, do), img_Con in dict_img_Con.items():
        img_Con_vis = pa.applyColorMap(img_Con, "RdBu_r", vmin, vmax)
        cv2.imwrite(f"fig_visualize_diff{d}_on{do}.png", img_Con_vis)

    for (d, dof), img_Coff in dict_img_Coff.items():
        img_Coff_vis = pa.applyColorMap(img_Coff, "RdBu_r", vmin, vmax)
        cv2.imwrite(f"fig_visualize_diff{d}_off{dof}.png", img_Coff_vis)


if __name__ == "__main__":
    main()
