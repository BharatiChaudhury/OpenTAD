import os
import json
import numpy as np
import random

FEATURE_DIR = "/srv/storage/stars@storage3.sophia.grid5000.fr/bchaudhu/projects/Introducing-Gating-and-Context-into-Temporal-Action-Detection/features_npy_v2_check"
OUTPUT_JSON = "mpii_tridet.json"

FPS = 30
STRIDE = 32

database = {}

files = [f for f in os.listdir(FEATURE_DIR) if f.endswith(".npy")]
files.sort()
print("Total files:", len(files))
random.seed(42)
random.shuffle(files)

split_idx = int(0.8 * len(files))

train_files = files[:split_idx]
val_files = files[split_idx:]

def process_file(fname, subset):

    video_id = fname.replace(".npy", "")

    parts = video_id.split("_")

    label = parts[3]

    feat = np.load(
        os.path.join(FEATURE_DIR, fname)
    )

    T = feat.shape[0]

    # -----------------------------------
    # Convert feature length to seconds
    # -----------------------------------

    duration = (T * STRIDE) / FPS

    return video_id, {
        "duration": float(duration),
        "frame": int(T), 
        "fps": FPS / STRIDE,
        "subset": subset,
        "annotations": [
            {
                "segment": [
                    0.0,
                    float(duration)
                ],
                "label": label
            }
        ]
    }

for f in train_files:

    vid, entry = process_file(f, "training")

    database[vid] = entry

for f in val_files:

    vid, entry = process_file(f, "validation")

    database[vid] = entry

output = {
    "database": database
}

with open(OUTPUT_JSON, "w") as f:

    json.dump(output, f, indent=4)

print("JSON created:", OUTPUT_JSON)
