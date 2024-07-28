import argparse
import subprocess
import os
import shutil
from nanobind.stubgen import StubGen


def green_print(msg):
    print("\033[92m" + msg + "\033[0m")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="Release", help="Configuration type")
    args = parser.parse_args()

    # Build nanobind module
    # Remove build directory
    shutil.rmtree("build", ignore_errors=True)

    # Run cmake and build
    ret1 = subprocess.run(["cmake", "-S", ".", "-B", "build"], shell=True)
    ret2 = subprocess.run(["cmake", "--build", "build", "--config", args.config], shell=True)
    if ret1.returncode != 0 or ret2.returncode != 0:
        raise Exception("Failed to build nanobind module")

    # In windows, copy build/Debug to src/eventellipsometry
    # and rename Debug to _eventellipsometry_impl
    if os.name == "nt":
        src = f"build/{args.config}"
        dst = "src/eventellipsometry/_eventellipsometry_impl"
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
    else:
        raise

    # Generate __init__.py
    with open("src/eventellipsometry/_eventellipsometry_impl/__init__.py", "w") as f:
        f.write("from ._eventellipsometry_impl import *")

    # Generate pyi file
    import eventellipsometry as ee

    module = ee._eventellipsometry_impl._eventellipsometry_impl
    sg = StubGen(module)
    sg.put(module)
    pyi = sg.get()
    with open("src/eventellipsometry/_eventellipsometry_impl/__init__.pyi", "w") as f:
        f.write(pyi)


if __name__ == "__main__":
    main()
