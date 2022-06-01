import os
import re
import json


def load_set(file_path):
    collection = set()
    with open(file_path, 'r') as file_:
        for line in file_:
            collection.add(line.rstrip())
    return collection


def load_txt(file_path):
    with open(file_path, encoding='utf-8') as f:
        data = [line.rstrip('\n') for line in f]
        return data


def load_dict(file_path, sep=",", skip_header=False):
    d = {}
    with open(file_path, encoding='utf-8') as f:
        if skip_header:
            f.readline()
        for line in f:
            data = re.split(sep, line.strip())
            d[data[0]] = data[1]
    return d


def load_csv(file_path):
    data = []
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            data.append(line.strip().split(','))
    return data


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def save_txt(file_path, data):
    with open(file_path, encoding='utf-8', mode='w+') as f:
        for item in data:
            f.write(str(item) + '\n')


def save_dict(file_path, data, sep=","):
    with open(file_path, "w", encoding="utf-8") as f:
        for k, v in data.items():
            f.write(f"{k}{sep}{v}\n")


def save_json(file_path, data):
    if not isinstance(data, list):
        data = [data]
    with open(file_path, 'w') as f:
        for item in data:
            strs = json.dumps(item)
            f.write(str(strs) + '\n')


def append_json(f, data):
    assert not f.closed
    assert f.mode.startswith("a")
    for item in data:
        strs = json.dumps(item)
        f.write(str(strs) + '\n')


def csv_to_json(csv_file, output_dir):
    data = load_dict(csv_file, sep=",", skip_header=True)
    filename = os.path.splitext(os.path.basename(csv_file))[0]
    save_json(os.path.join(output_dir, filename + ".json"), data)


def json_to_csv(json_file, output_dir):
    data = load_json(json_file)
    filename = os.path.splitext(os.path.basename(json_file))[0]
    output_file = os.path.join(output_dir, filename + ".csv")
    save_dict(output_file, data)
