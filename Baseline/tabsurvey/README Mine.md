To run a single model on a single dataset call:

``python train.py --config/<config-file of the dataset>.yml --model_name <Name of the Model>``

All parameters set in the config file, can be overwritten by command line arguments, for example:

- ``--optimize_hyperparameters`` Uses [Optuna](https://optuna.org/) to run a hyperparameter optimization. If not set, the parameters listed in the `best_params.yml` file are used.

- ``--n_trails <number trials>`` Number of trials to run for the hyperparameter search

- ``--epochs <number epochs>`` Max number of epochs

- ``--use_gpu`` If set, available GPUs are used (specified by `gpu_ids`)

- ... and so on. All possible parameters can be found in the config files or calling: 
``python train.y -h``


### Run multiple models on multiple datasets

use testdji.sh 