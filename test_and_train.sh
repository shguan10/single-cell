#!/bin/bash

echo "Training first stage\n"
python staged_simple_training.py
echo "Training second stage\n"
python stage_2_training.py
echo "Testing overall model\n"
python overall_staged_testing.py
