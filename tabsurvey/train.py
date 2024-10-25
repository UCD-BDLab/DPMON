import logging
import sys
import json

import optuna

from models import str2model
from utils.load_data import load_data
from utils.scorer import get_scorer
from utils.timer import Timer
from utils.io_utils import save_results_to_file, save_hyperparameters_to_file, save_loss_to_file
from utils.parser import get_parser, get_given_parameters_parser

from sklearn.model_selection import KFold, StratifiedKFold  # , train_test_split

def save_results_to_json(args, results, train_time, inference_time, model_params):
    file_path = f"results3_{args.model_name}_{args.dataset}.json"
    
    # Create a dictionary of all the important data to save
    data = {
        "results": results,
        "train_time": train_time,
        "inference_time": inference_time,
        "model_params": model_params
    }
    
    # Load existing data, update it, and save it back
    try:
        with open(file_path, 'r+') as file:  # Open the file in read/write mode
            file_data = json.load(file)       # Load existing data
            file_data.append(data)            # Append new data
            file.seek(0)                      # Move to the start of the file
            json.dump(file_data, file, indent=4)
    except FileNotFoundError:
        with open(file_path, 'w') as file:  # If file does not exist, create it and write the data
            json.dump([data], file, indent=4)  # Save as a list of results

def cross_validation(model, X, y, args, save_model=False):
    # Record some statistics and metrics
    sc = get_scorer(args)
    train_timer = Timer()
    test_timer = Timer()

    if args.objective == "regression":
        kf = KFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seed)
    elif args.objective == "classification" or args.objective == "binary":
        kf = StratifiedKFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seed)
    else:
        raise NotImplementedError("Objective" + args.objective + "is not yet implemented.")

    
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=args.seed)

        # Create a new unfitted version of the model
        curr_model = model.clone()

        # Train model
        train_timer.start()
        loss_history, val_loss_history = curr_model.fit(X_train, y_train, X_test, y_test)  # X_val, y_val)
        train_timer.end()

        # Test model
        test_timer.start()
        curr_model.predict(X_test)
        test_timer.end()

        # Save model weights and the truth/prediction pairs for traceability
        curr_model.save_model_and_predictions(y_test, i)

        if save_model:
            save_loss_to_file(args, loss_history, "loss", extension=i)
            save_loss_to_file(args, val_loss_history, "val_loss", extension=i)

        # Compute scores on the output
        sc.eval(y_test, curr_model.predictions, curr_model.prediction_probabilities)

        print(sc.get_results())

    # Best run is saved to file
    if save_model:
        print("Results:", sc.get_results())
        print("Train time:", train_timer.get_average_time())
        print("Inference time:", test_timer.get_average_time())

        # Save all statistics to a JSON file
        save_results_to_json(args, sc.get_results(), train_timer.get_average_time(), test_timer.get_average_time(), model.params)

        # Save the all statistics to a file
        save_results_to_file(args, sc.get_results(),
                             train_timer.get_average_time(), test_timer.get_average_time(),
                             model.params)

    # print("Finished cross validation")
    return sc, (train_timer.get_average_time(), test_timer.get_average_time())


class Objective(object):
    def __init__(self, args, model_name, X, y):
        # Save the model that will be trained
        self.model_name = model_name

        # Save the trainings data
        self.X = X
        self.y = y

        self.args = args

    def __call__(self, trial):
        # Define hyperparameters to optimize
        trial_params = self.model_name.define_trial_parameters(trial, self.args)

        # Create model
        model = self.model_name(trial_params, self.args)

        # Cross validate the chosen hyperparameters
        sc, time = cross_validation(model, self.X, self.y, self.args)

        save_hyperparameters_to_file(self.args, trial_params, sc.get_results(), time)

        return sc.get_objective_result()


def main(args):
    print("Start hyperparameter optimization")
    X, y = load_data(args)
        
    # print(f"X type: {type(X)}")
    # print(f"y type: {type(y)}")
    # print(f"X shape: {X.shape}")
    # print(f"y shape: {y.shape}")
    # print(f"X value for data: {X}")
    # print(f"y value for data: {y}")

    model_name = str2model(args.model_name)
    params_file = f"best_params3_{args.model_name}_{args.dataset}.json"  
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    if args.dataset.startswith("Current"):
        num_iterations = 10
    elif args.dataset.startswith("Former"):
        num_iterations = 5
    else:
        num_iterations = 5

    try:
        with open(params_file, 'r') as file:
            best_params = json.load(file)

    except FileNotFoundError:
        study_name = args.model_name + "_" + args.dataset
        storage_name = "sqlite:///{}.db".format(study_name)

        study = optuna.create_study(direction=args.direction,
                                    study_name=study_name,
                                    storage=storage_name,
                                    load_if_exists=True)
        study.optimize(Objective(args, model_name, X, y), n_trials=args.n_trials)
        best_params = study.best_trial.params

        with open(params_file, 'w') as file:
            json.dump(best_params, file)

    print("Best parameters:", best_params)

    for _ in range(num_iterations):
        model = model_name(best_params, args)
        cross_validation(model, X, y, args, save_model=True)


    # study_name = args.model_name + "_" + args.dataset
    # storage_name = "sqlite:///{}.db".format(study_name)

    # study = optuna.create_study(direction=args.direction,
    #                             study_name=study_name,
    #                             storage=storage_name,
    #                             load_if_exists=True)
    # study.optimize(Objective(args, model_name, X, y), n_trials=args.n_trials)
    # print("Best parameters:", study.best_trial.params)

    # # Run best trial again and save it!
    # model = model_name(study.best_trial.params, args)
    # cross_validation(model, X, y, args, save_model=True)


def main_once(args):
    print("Train model with given hyperparameters")
    X, y = load_data(args)
    
    print(f"X type: {type(X)}")
    print(f"y type: {type(y)}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X value for data: {X}")
    print(f"y value for data: {y}")

    model_name = str2model(args.model_name)

    parameters = args.parameters[args.dataset][args.model_name]
    model = model_name(parameters, args)

    sc, time = cross_validation(model, X, y, args)
    print(sc.get_results())
    print(time)


if __name__ == "__main__":
    parser = get_parser()
    arguments = parser.parse_args()
    print(arguments)

    if arguments.optimize_hyperparameters:
        main(arguments)
    else:
        # Also load the best parameters
        parser = get_given_parameters_parser()
        arguments = parser.parse_args()
        main_once(arguments)
