#!/bin/bash

# Run the Former Smokers pipeline
./testFormer.sh

# Run the Current Smokers pipeline
./testCurrent.sh

echo "Both pipelines have completed successfully."
