# import json
# import random
# import csv

# metadata_train_file = "/mnt/LS226/LS25/rahul2022387/OpenSLR/Dataset/train/metadata.csv"
# metadata_test_file = "/mnt/LS226/LS25/rahul2022387/OpenSLR/Dataset/test/metadata.csv"

# output_train_path = "/home/rahul_b/TTS/VoiceCraft/datasets/mucs/manifests/metadata_mucs_train.jsonl"
# output_test_path ="/home/rahul_b/TTS/VoiceCraft/datasets/mucs/manifests/metadata_mucs_test.jsonl"


# with open(metadata_train_file, "r") as f:
#     reader = csv.DictReader(f)
#     data = list(reader)

# with open(output_train_path, "w") as f_train:
#     for entry in data:
#         json_line = json.dumps(entry)
#         f_train.write(json_line + "\n")
     


# with open(metadata_test_file, "r") as f:
#     reader = csv.DictReader(f)
#     data = list(reader)

# with open(output_test_path, "w") as f_test:
#     for entry in data:
#         json_line = json.dumps(entry)
#         f_test.write(json_line + "\n")
import json
import random
import csv

metadata_train_file = "/mnt/LS226/LS25/rahul2022387/OpenSLR/Dataset/train/metadata.csv"
metadata_test_file = "/mnt/LS226/LS25/rahul2022387/OpenSLR/Dataset/test/metadata.csv"

output_train_path = "/home/rahul_b/TTS/VoiceCraft/datasets/mucs/manifests/metadata_mucs_train.jsonl"
output_test_path = "/home/rahul_b/TTS/VoiceCraft/datasets/mucs/manifests/metadata_mucs_test.jsonl"

# Process train metadata
with open(metadata_train_file, "r") as f:
    reader = csv.DictReader(f)
    data = list(reader)

with open(output_train_path, "w") as f_train:
    for entry in data:
        if "audio file" in entry:
            entry["audiofilepath"] = entry.pop("audio file")
        json_line = json.dumps(entry)
        f_train.write(json_line + "\n")

# Process test metadata
with open(metadata_test_file, "r") as f:
    reader = csv.DictReader(f)
    data = list(reader)

with open(output_test_path, "w") as f_test:
    for entry in data:
        if "audio file" in entry:
            entry["audiofilepath"] = entry.pop("audio file")
        json_line = json.dumps(entry)
        f_test.write(json_line + "\n")
