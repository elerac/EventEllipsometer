import argparse
from pathlib import Path
import re
import cv2
from tqdm import trange


# sort by filename (numerically)
def sort_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s.name)]


def main():
    parser = argparse.ArgumentParser(description="Create a video from images.")
    parser.add_argument("dir", help="Directory containing images.")
    parser.add_argument("--fps", type=float, default=30, help="Frames per second.")
    args = parser.parse_args()

    print("Images to video")
    print("Source:", args.dir)

    filename_list = sorted(Path(args.dir).glob("*.png"), key=sort_key)
    filename_list = [str(path) for path in filename_list]
    print("Number of images:", len(filename_list))
    if len(filename_list) == 0:
        print("No images found.")
        return

    for i, filename in enumerate(filename_list):
        print(f"{i}: {Path(filename).name}")

    img_0 = cv2.imread(filename_list[0], 1)
    height, width, _ = img_0.shape
    size = (width, height)

    # write with mp4
    path_dst = Path(args.dir) / "video.mp4"
    print("Destination:", path_dst)
    print("  FPS:", args.fps)
    print("  Size:", size)
    print("  Length:", len(filename_list))
    out = cv2.VideoWriter(str(path_dst), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size)
    for i in trange(len(filename_list)):
        img = cv2.imread(filename_list[i], 1)
        out.write(img)

    out.release()

    print("Done")


if __name__ == "__main__":
    main()
