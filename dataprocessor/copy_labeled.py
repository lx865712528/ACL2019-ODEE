import json
import os
import sys

import yaml


def input_queue(INPUT_DIR):
    for DATE_STR in os.listdir(INPUT_DIR):
        if DATE_STR.startswith("."): continue
        for ID in os.listdir(os.path.join(INPUT_DIR, DATE_STR)):
            if ID.startswith("."): continue
            file_path = os.path.join(INPUT_DIR, DATE_STR, ID, "key.txt")
            if os.path.exists(file_path):
                yield file_path


def copy_labeled(FILE_PATH):
    name = "-".join(FILE_PATH.split(os.sep)[-3:-1])
    events = []
    event = None
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            elif line.startswith("COMMENTS") or \
                    line.startswith("TENSE") or \
                    line.startswith("CAUSE") or \
                    line.startswith("TRIGGER"):
                continue
            elif line.startswith("EVENT"):
                if not event is None:
                    events.append(event)
                event = {}
            else:
                key_pos = line.find(":")
                key = line[:key_pos]
                l_pos = line.find("{")
                r_pos = line.rfind("}")
                values = line[l_pos + 1:r_pos].split(", ") if l_pos + 1 < r_pos else []
                event[key] = values
    if not (event is None or len(event.keys()) == 0):
        events.append(event)
    return name, events


def store_labeled(events, OUTPUT_PATH):
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, sort_keys=True, ensure_ascii=False)
    print("STORED %s" % OUTPUT_PATH)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("PYTHON odee_preprocess.py INPUT_DIR OUTPUT_DIR")
        sys.exit(-1)
    INPUT_DIR, OUTPUT_DIR = sys.argv[1], sys.argv[2]
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    with open("setting.yaml", "r") as stream:
        all_setting = yaml.load(stream)
    print(INPUT_DIR, OUTPUT_DIR)

    for file_path in input_queue(INPUT_DIR):
        print("Processing %s" % file_path)
        name, events = copy_labeled(file_path)
        # dump cleaned stanford json
        json_output_path = os.path.join(OUTPUT_DIR, name + ".json.labeled")
        store_labeled(events, json_output_path)
    print("Done!")
