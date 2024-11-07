#!/bin/bash

# Run SAINT for both former and current smokers
./tabsurvey/testFormer.sh
./tabsurvey/testCurrent.sh

# Run baseline.py for RandomForestClassifier and LogisticRegression benchmarks
python baseline.py

echo "All tests and benchmarks have completed successfully."
