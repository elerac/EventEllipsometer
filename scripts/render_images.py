import mitsuba as mi
import drjit as dr

mi.set_variant("cuda_spectral_polarized")  # The CUDA mode may takes few time to start up but faster renderings
# mi.set_variant("scalar_spectral_polarized")

import argparse
import time
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import shutil
import math

import numpy as np
from tqdm import tqdm

import polanalyser as pa


def pprojector(image: np.ndarray, fov: float, scale: float, origin: list[float], target: list[float], up: list[float], angle: float, delta: float = 10) -> dict:
    """Create a mitsuba3 scene of a projector attached with a polarizer in python dict format."""
    # Get image size
    bitmap = mi.Bitmap(image)
    width, height = bitmap.size()

    # Scaling of polarizer
    x_scale = delta * math.tan(math.radians(fov) / 2.0)
    y_scale = x_scale * height / width

    # Transform
    t_look_at = mi.ScalarTransform4f.look_at(origin=origin, target=target, up=up)
    t_translate = mi.ScalarTransform4f.translate([0, 0, delta])
    t_scale = mi.ScalarTransform4f.scale([x_scale, y_scale, 1.0])

    projector = {
        "projector": {
            "type": "projector",
            "irradiance": {"type": "bitmap", "bitmap": bitmap, "filter_type": "nearest"},
            "scale": scale,
            "fov": fov,
            "to_world": t_look_at,
        }
    }

    polarizer = {
        "polarizer_projector": {
            "type": "rectangle",
            # "flip_normals": True,
            "to_world": t_look_at @ t_translate @ t_scale,
            "polarizer": {"type": "polarizer", "theta": {"type": "spectrum", "value": 0}},
        },
    }

    qwp = {
        "qwp_projector": {
            "type": "rectangle",
            # "flip_normals": True,
            "to_world": t_look_at @ t_translate @ t_translate @ t_scale,
            "retarder": {"type": "retarder", "theta": {"type": "spectrum", "value": angle}},
        },
    }

    return {**projector, **polarizer, **qwp}


def run_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str, help="Output file path")
    parser.add_argument("values", type=float, nargs="+", help="Values to fill the images")
    args, _ = parser.parse_known_args()

    # Show date and time and arguments
    print(f"Date and time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"    Arguments: output={args.output}, values[0]={args.values[0]:.3f}, values[-1]={args.values[-1]:.3f}, len(values)={len(args.values)}")

    pbrdf_blue_biliard = mi.load_dict(
        {
            "type": "measured_polarized",
            "filename": "scenes/pbsdf/9_blue_billiard_mitsuba/9_blue_billiard_inpainted.pbsdf",
            "alpha_sample": 0.2,
        }
    )

    pbrdf_ochar_silicone = mi.load_dict(
        {
            "type": "measured_polarized",
            "filename": "scenes/pbsdf/20_ocher_silicone_mitsuba/20_ocher_silicon_inpainted.pbsdf",
            "alpha_sample": 0.2,
        }
    )

    def load_scene(theta=0, t=0):
        # theta = 0
        image = np.full((2, 2, 3), 255, dtype=np.uint8)
        scene = mi.load_dict(
            {
                "type": "scene",
                "integrator": {
                    "type": "stokes",
                    "nested": {"type": "volpath", "max_depth": -1},
                },
                "sensor": {
                    "type": "perspective",
                    "to_world": mi.ScalarTransform4f.look_at(origin=[0, 0, 50], target=[0, 0, 0], up=[0, 1, 0]),
                    # "to_world": mi.ScalarTransform4f.look_at(origin=[0, 0, 1], target=[0, 0, 0], up=[0, 1, 0]),
                    # "to_world": mi.ScalarTransform4f.look_at(origin=[200, 100, 600], target=[100, 0, 500], up=[0, 1, 0]),
                    "fov": 4,
                    # "fov": 100,
                    "film": {"type": "hdrfilm", "width": 346 // 4, "height": 260 // 4},
                },
                # "constant_emitter": {
                #     "type": "constant",
                #     "radiance": {
                #         "type": "spectrum",
                #         "value": 1,
                #     },
                # },
                # "sphere1": {
                #     "type": "sphere",
                #     "center": [0, 0, 0],
                #     "radius": 1.0,
                #     "to_world": mi.ScalarTransform4f.rotate([1, 0, 0], 90),
                #     # "bsdf": {
                #     #     "type": "measured_polarized",
                #     #     "filename": "pbsdf/9_blue_billiard_mitsuba/9_blue_billiard_inpainted.pbsdf",
                #     #     "alpha_sample": 0.2,
                #     # },
                #     "bsdf": {
                #         "type": "blendbsdf",
                #         "weight": {
                #             "type": "checkerboard",
                #             "color0": 0.0,
                #             "color1": 1.0,
                #             "to_uv": mi.ScalarTransform4f.scale(3),
                #         },
                #         "bsdf_0": {
                #             "type": "measured_polarized",
                #             "filename": "pbsdf/20_ocher_silicone_mitsuba/20_ocher_silicon_inpainted.pbsdf",
                #             "alpha_sample": 0.2,
                #         },
                #         "bsdf_1": {
                #             "type": "measured_polarized",
                #             "filename": "pbsdf/19_mint_silicone_mitsuba/19_mint_silicon_inpainted.pbsdf",
                #             "alpha_sample": 0.2,
                #         },
                #     },
                # },
                "plane1": {
                    "type": "rectangle",
                    "to_world": mi.ScalarTransform4f.translate([0, 0, -10]) @ mi.ScalarTransform4f.scale(5),
                    "bsdf": pbrdf_ochar_silicone,
                },
                # "plane2": {
                #     "type": "rectangle",
                #     "to_world": mi.ScalarTransform4f.translate([0, -3, 0]) @ mi.ScalarTransform4f.rotate([1, 0, 0], 85) @ mi.ScalarTransform4f.scale(5),
                #     "bsdf": {
                #         "type": "measured_polarized",
                #         "filename": "pbsdf/15_white_silicone_mitsuba/15_white_silicon_inpainted.pbsdf",
                #         "alpha_sample": 0.2,
                #     },
                # },
                "bunny": {
                    "type": "ply",
                    "filename": "scenes/meshes/bunny.ply",
                    "to_world": mi.ScalarTransform4f.translate([0, -0.25, 0]) @ mi.ScalarTransform4f.scale(14),
                    # "to_world": mi.ScalarTransform4f.translate([(t - 0.5) * 0.5, 0, 0]) @ mi.ScalarTransform4f.translate([0, -0.25, 0]) @ mi.ScalarTransform4f.scale(14),
                    "bsdf": pbrdf_blue_biliard,
                },
            }
            | pprojector(image, 90, 1000000, [100, 0, 500], [0, 0, 0], [0, 1, 0], np.rad2deg(theta), delta=50)
        )
        return scene

    scene = load_scene()
    # params = mi.traverse(scene)
    # print(params)

    mi.set_log_level(mi.LogLevel.Error)

    time_min = 0
    time_max = 1 / 60  # * #20
    # num = 180 * 50  # * 20  # // 10000
    img_list = []
    mm_psa_list = []
    mm_psg_list = []
    theta_list = []
    for i, t in enumerate(tqdm(args.values)):
        t_abs = t * (time_max - time_min) + time_min
        rot_speed = 60 * np.pi  # [rad/s]
        theta = t_abs * rot_speed  # % (2 * np.pi)

        scene = load_scene(theta, t)

        # params = mi.traverse(scene)

        # if i == 0:
        #     bunny_vpos_init = np.array(params["bunny.vertex_positions"])

        # bunny_vpos = np.array(params["bunny.vertex_positions"])
        # bunny_vpos = np.zeros_like(bunny_vpos, dtype=np.float32)
        # bunny_vpos[0::3] = (t - 0.5) * 0.5  # x
        # # bunny_vpos[1::3] = 0  # y
        # # bunny_vpos[2::3] = 0  # z

        # is_cuda = "cuda" in mi.variant()
        # if is_cuda:
        #     bunny_vpos = dr.cuda.Float(bunny_vpos + bunny_vpos_init)
        # else:
        #     bunny_vpos = dr.scalar.ArrayXf(bunny_vpos + bunny_vpos_init)

        # params["bunny.vertex_positions"] = bunny_vpos
        # params["qwp_projector.bsdf.theta.value"] = np.rad2deg(theta)
        # params.update()

        image = mi.render(scene, spp=256)

        # Conver bitmap to ndarray
        bitmap = mi.Bitmap(image, channel_names=["R", "G", "B"] + scene.integrator().aov_names())
        channels = dict(bitmap.split())
        width, height = bitmap.size()
        img_rgb_stokes = np.empty((height, width, 3, 4))
        for si in range(4):
            img_rgb_stokes[..., si] = channels[f"S{si}"]
        img_bgr_stokes = img_rgb_stokes[..., ::-1, :]

        # img_bgr = img_bgr_stokes[..., 0]
        # img_stokes = img_bgr_stokes[..., 1, :]
        # img_aolp = pa.cvtStokesToAoLP(img_stokes)
        # img_dolp = pa.cvtStokesToDoLP(img_stokes)
        # img_docp = pa.cvtStokesToDoCP(img_stokes)
        # img_dop = pa.cvtStokesToDoP(img_stokes)
        # img_ellipticity_angle = pa.cvtStokesToEllipticityAngle(img_stokes)
        # img_bgr_vis = np.clip(img_bgr ** (1 / 2.2) * 255, 0, 255).astype(np.uint8)
        # img_aolp_vis = pa.applyColorToAoLP(img_aolp)
        # img_top_vis = pa.applyColorToToP(img_ellipticity_angle, img_dop)
        # img_cop_vis = pa.applyColorToCoP(img_ellipticity_angle, img_docp)
        # img_dop_vis = pa.applyColorToDoP(img_dop)
        # img_dolp_vis = pa.applyColorToDoP(img_dolp)
        # img_docp_vis = pa.applyColorToDoP(img_docp)
        # img_grid = pa.makeGrid([img_bgr_vis, img_aolp_vis, img_top_vis, img_cop_vis, img_dop_vis, img_dolp_vis, img_docp_vis], 2)
        # img_grid = cv2.resize(img_grid, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
        # cv2.putText(img_grid, f"theta: {np.rad2deg(theta):.1f} [deg]", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.imshow("grid", img_grid)
        # cv2.waitKey(1)

        # img_bgr_stokes = render(theta)  # (height, width, 3, 4)
        mm_psg = pa.qwp(theta) @ pa.polarizer(0)  # (4, 4)
        mm_psa = pa.polarizer(0) @ pa.qwp(5 * theta)  # (4, 4)
        img_bgr = (mm_psa @ img_bgr_stokes[..., None]).squeeze()[..., 0]
        # pcontainer.append(img_bgr.astype(np.float32), mm_psa, mm_psg, theta=theta)

        img_list.append(img_bgr)
        mm_psa_list.append(mm_psa)
        mm_psg_list.append(mm_psg)
        theta_list.append(theta)

    pa.imwriteMultiple(args.output, img_list, mm_psa=mm_psa_list, mm_psg=mm_psg_list, theta=theta_list)


def run_coordinator():
    time_start = time.time()
    python = sys.executable
    filename_py = __file__

    num = 180 * 100  # * 20
    num = 100
    chunk_size = 1000

    print(f"Start running Mitsuba3 renderer. {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python executable: {python}")
    print(f"Python script: {filename_py}")
    print(f"Number of images: {num}")
    print(f"Chunk size: {chunk_size}")
    print("-" * 80)

    num_split = np.ceil(num / chunk_size).astype(int)
    t = np.linspace(0, 1, num)
    with ThreadPoolExecutor(max_workers=1) as executor:
        for i in range(num_split):
            values = t[i * chunk_size : min((i + 1) * chunk_size, num)]
            values_str = " ".join([str(val) for val in values])
            cmd = [python, filename_py, f"output{i}", "--worker"] + values_str.split()
            try:
                executor.submit(subprocess.run, cmd).result()
            except Exception as e:
                print(f"Error occurred: {e}")
                sys.exit(1)

    print("All workers have finished.")

    # Gather the images
    imlist = []
    props = {}
    for i in range(num_split):
        images_i, props_i = pa.imreadMultiple(f"output{i}")
        # Delete the folder "output{i}" and its contents
        shutil.rmtree(f"output{i}")

        imlist = imlist + images_i.tolist()
        # for all keys in props
        for key in props_i.keys():
            if key in props:
                props[key] = props[key] + props_i[key].tolist()
            else:
                props[key] = props_i[key].tolist()

        # print(images)

    print("All images have been gathered.")
    print("Number of images: ", len(imlist))
    print("Keys in props: ", props.keys())

    pa.imwriteMultiple("rendered/bunny", np.array(imlist), **props)

    print(f"Total elapsed time: {(time.time() - time_start) / 3600:.2f} [h]")


def main():
    # This script renders sequences of images with Mitsuba3.
    # I divide the rendering process into two parts: coordinator and worker.
    # The worker: renders several images and saves them to a file.
    # The coordinator: manages the workers and gathers the images.

    # This separation is awkward, but it is necessary to avoid weird behaviors of Mitsuba3.
    # I found that Mitsuba3 stops rendering when the number of images exceeds a certain number.

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help="Run as a worker")
    args, _ = parser.parse_known_args()

    if not args.worker:
        run_coordinator()
    else:
        run_worker()


if __name__ == "__main__":
    main()
