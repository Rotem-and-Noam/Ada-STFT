import os

ckpt_dir = ["../checkpoints2"]
save_files = ["ckpt_last.pt", "ckpt_best.pt"]

for dir in ckpt_dir:
    sub_folders = os.listdir(dir)
    for sub in sub_folders:
        dir_path = os.path.join(dir, sub)
        files = os.listdir(dir_path)
        for file in files:
            if file not in save_files:
                path = os.path.join(dir_path, file)
                os.remove(path)
