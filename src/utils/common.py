import os
import sys
import logging
import datetime
import subprocess

from hashlib import sha256


def gen_sha256(data):
    return sha256(str(data).encode()).hexdigest()


def get_current_time_str():
    now = datetime.datetime.now()
    return now.isoformat()


def multi_makedirs(dirs, exist_ok=False):
    if not isinstance(dirs, list):
        dirs = list(dirs)
    for d in dirs:
        os.makedirs(d, exist_ok=exist_ok)


def get_files_multilevel(root, pattern):
    list_files = []
    for root, _, files in os.walk(root):
        for f in files:
            if pattern in f:
                list_files.append(os.path.join(root, f))
    return list_files


def get_file_logger(file_path):
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(file_path),
            logging.StreamHandler(sys.stdout)
        ])
    return logging.getLogger()


def run_command(command):
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    return p.communicate()
