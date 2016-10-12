#!/bin/bash
python src/model.py tr cfg/model2.json
python src/test.py cfg/model2.json weights/linear_regression.weights linear_regression.csv
