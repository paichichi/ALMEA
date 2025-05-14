#!/bin/bash

echo "Starting FBDB15K experiments"
bash run_experiments.sh 0 FBDB15K 0.2 0.45
#bash run_experiments.sh 0 FBDB15K 0.5 0.45
#bash run_experiments.sh 0 FBDB15K 0.8 0.45

echo "Starting FBYG15K experiments"
#bash run_experiments.sh 0 FBYG15K 0.2 0.50
#bash run_experiments.sh 0 FBYG15K 0.5 0.50
#bash run_experiments.sh 0 FBYG15K 0.8 0.50
echo "All experiments completed."
