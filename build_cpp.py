import subprocess
import os
import shutil


def main():
    # Build nanobind module
    ret1 = subprocess.run(["cmake", "-S", ".", "-B", "build"], shell=True)
    ret2 = subprocess.run(["cmake", "--build", "build"], shell=True)
    if ret1.returncode != 0 or ret2.returncode != 0:
        raise Exception("Failed to build nanobind module")

    # In windows, copy build/Debug to src/eventellipsometry
    # and rename Debug to _eventellipsometry_impl
    if os.name == "nt":
        shutil.rmtree("src/eventellipsometry/_eventellipsometry_impl", ignore_errors=True)
        shutil.copytree("build/Debug", "src/eventellipsometry/_eventellipsometry_impl")
    else:
        raise


if __name__ == "__main__":
    main()
