import os

DIVISIBLE_BY = 200

def find_all_checkpoint_dirs(directory):
    checkpoint_dirs = []
    for filename in os.listdir(directory):
        if filename.startswith("checkpoints_") and os.path.isdir(os.path.join(directory, filename)):
            checkpoint_dirs.append(os.path.join(directory, filename))
    return checkpoint_dirs

def find_all_checkpoints(directory):
    checkpoints = []
    for filename in os.listdir(directory):
        if filename.startswith("ckpt_") and filename.endswith(".pt"):
            checkpoints.append(os.path.join(directory, filename))
    return checkpoints

def filter_checkpoints(checkpoints):
    return [ckpt for ckpt in checkpoints if int(os.path.basename(ckpt).split('_')[-1].split('.')[0]) % DIVISIBLE_BY != 0]

def delete_checkpoints(checkpoints):
    for ckpt in checkpoints:
        os.remove(ckpt)
        print(f"Deleted checkpoint: {ckpt}")

if __name__ == "__main__":
    checkpoint_dirs = find_all_checkpoint_dirs(".")
    for checkpoint_dir in checkpoint_dirs:
        checkpoints = find_all_checkpoints(checkpoint_dir)
        checkpoints_to_delete = filter_checkpoints(checkpoints)
        delete_checkpoints(checkpoints_to_delete)