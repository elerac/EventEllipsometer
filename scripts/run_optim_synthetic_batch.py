import subprocess

# Loop for different max_iter_svd values
for i in range(11):
    subprocess.run(["rye", "run", "python", "./optimize_video.py", "--filename_raw", "./rendered/bunny5_masked.npz", "--max_iter_svd", str(i), "--max_iter_propagate", "0"])

# Loop for different max_iter_propagate values
for j in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    subprocess.run(["rye", "run", "python", "./optimize_video.py", "--filename_raw", "./rendered/bunny5_masked.npz", "--max_iter_svd", "5", "--max_iter_propagate", str(j)])
