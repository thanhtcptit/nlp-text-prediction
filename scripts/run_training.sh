#!/bin/bash

python run.py train configs/bert/albert-b_fl.json -f
python run.py train configs/bert/bert-b_fl.json -f
python run.py train configs/bert/bert-l_fl.json -f
python run.py train configs/bert/bert-tw_fl.json -f