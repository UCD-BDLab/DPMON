#!/bin/bash

#I will first use 10 trials to find the best possible parameters
N_TRIALS=100
EPOCHS=10

# Models and configurations
declare -A MODELS
MODELS=(
         ["SAINT"]="torch"
          )

CONFIGS=( "config/current.yml" )


# Network configurations
CurrentSmokers_Networks=(
    "CurrentSmokersNetwork1-98"
    "CurrentSmokersNetwork2-35"
    "CurrentSmokersNetwork3-16"
    "CurrentSmokersNetwork4-12"
    "CurrentSmokersNetwork5-12"
)

# --use_gpu

# Loop through each configuration, model, and network
for config in "${CONFIGS[@]}"; do
    for model in "${!MODELS[@]}"; do
        for network in "${CurrentSmokers_Networks[@]}"; do
            # Extract the number of features from the network name
            num_features="${network##*-}"

            printf "\n\n----------------------------------------------------------------------------\n"
            printf 'Training %s with %s on dataset %s (features: %s) in env %s\n\n' "$model" "$config" "$network" "$num_features" "${MODELS[$model]}"

            # Loop only executes this line 10 times!
            # 1 for the CONFIG file, 2 for the Models (SAINT and TabTransformer) and 5 times for each network
            python train.py --config "$config" --model_name "$model" --dataset "$network" --num_features "$num_features" --n_trials $N_TRIALS --epochs $EPOCHS --optimize_hyperparameters --use_gpu
        done
    done
done
