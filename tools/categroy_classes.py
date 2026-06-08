import json

JSON_PATH = "mpii_tridet.json"
OUTPUT_PATH = "category_idx.txt"

# -----------------------------------------
# LOAD JSON
# -----------------------------------------

with open(JSON_PATH, "r") as f:
    data = json.load(f)

database = data["database"]

# -----------------------------------------
# COLLECT LABELS
# -----------------------------------------

labels = set()

for video_id, video_data in database.items():

    anns = video_data["annotations"]

    for ann in anns:

        labels.add(ann["label"])

# -----------------------------------------
# SORT LABELS
# -----------------------------------------

labels = sorted(list(labels))

# -----------------------------------------
# WRITE category_idx.txt
# -----------------------------------------

with open(OUTPUT_PATH, "w") as f:

    for idx, label in enumerate(labels):

        f.write(f"{idx} {label}\n")

print("Created:", OUTPUT_PATH)

print("\nClasses:")

for idx, label in enumerate(labels):

    print(idx, label)