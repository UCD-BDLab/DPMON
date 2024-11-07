import os
from args import *
from dataset import *
from model import *
import statistics
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from torch_geometric.transforms import RandomNodeSplit
from torchmetrics.classification import MulticlassAUROC

from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import tempfile
from ray import train, tune
from ray.train import Checkpoint

# Hyperparameters for the Entire Neural Network Pipeline
PIPELINE_CONFIGS = {
    'gnn_layer_num': tune.choice([2, 4, 8, 16, 32, 64, 128]),
    'gnn_hidden_dim': tune.choice([4, 8, 16, 32, 64, 128]),
    'lr': tune.loguniform(1e-4, 1e-1),
    'weight_decay': tune.loguniform(1e-4, 1e-1),
    'nn_hidden_dim1': tune.choice([4, 8, 16, 32, 64, 128]),
    'nn_hidden_dim2': tune.choice([4, 8, 16, 32, 64, 128]),
    'num_epochs': tune.choice([2, 16, 64, 512, 1024, 4096, 8192]),
}
HYPERPARAMETERS_TUNING = False
GNN_MODEL = 'GIN'

class NeuralNetwork(nn.Module):
    def __init__(self,
                 gnn_input_dim,
                 gnn_hidden_dim,
                 gnn_layer_num,
                 ae_encoding_dim,
                 nn_input_dim,
                 nn_hidden_dim1,
                 nn_hidden_dim2,
                 nn_output_dim):
        super(NeuralNetwork, self).__init__()

        self.gnn = ''  # GNN Model
        if GNN_MODEL == 'GCN':
            self.gnn = GCN(input_dim=gnn_input_dim, hidden_dim=gnn_hidden_dim, layer_num=gnn_layer_num)
        elif GNN_MODEL == 'GAT':
            self.gnn = GAT(input_dim=gnn_input_dim, hidden_dim=gnn_hidden_dim, layer_num=gnn_layer_num)
        elif GNN_MODEL == 'GIN':
            self.gnn = GIN(input_dim=gnn_input_dim, hidden_dim=gnn_hidden_dim, layer_num=gnn_layer_num)
        elif GNN_MODEL == 'SAGE':
                self.gnn = SAGE(input_dim=gnn_input_dim, hidden_dim=gnn_hidden_dim, layer_num=gnn_layer_num)
        else:
            raise RuntimeError(f'GNN model {GNN_MODEL} not Recognized')

        # TODO: Revisit the Autoencoder - Since we Only Need to Encode so the Model doesn't learn here!
        self.autoencoder = Autoencoder(input_dim=gnn_hidden_dim, encoding_dim=ae_encoding_dim)
        self.dim_averaging = DimensionAveraging()
        self.predictor = DownstreamTaskNN(nn_input_dim, nn_hidden_dim1, nn_hidden_dim2, nn_output_dim)
        # self.predictor = LogisticRegression(nn_input_dim, nn_output_dim)

    def forward(self, omics_dataset, omics_network_tg):
        # print(omics_network_tg)
        omics_network_nodes_embedding = self.gnn(omics_network_tg)

        # print("Dimenions of Embeddings after GCN")
        # print(omics_network_nodes_embedding.shape)
        # print(omics_network_nodes_embedding)
        omics_network_nodes_embedding_ae = self.autoencoder(omics_network_nodes_embedding)
        # print("Scaling Factors %s" % omics_network_nodes_embedding_ae)
        # print("Embeddings %s" % omics_network_nodes_embedding_ae)

        # print("Dimensions of Embeddings after Autoencoder")
        # print(omics_network_nodes_embedding_ae.shape)
        # print(omics_network_nodes_embedding_ae)

        omics_network_nodes_embedding_avg = self.dim_averaging(omics_network_nodes_embedding_ae)

        # print("Dimensions of Embeddings after Averaging")
        # print(omics_network_nodes_embedding_avg.shape)
        # print("Averaging Embeddings %s" % omics_network_nodes_embedding_avg)

        # Multiplying the original_dataset with the generated embeddings
        omics_dataset_with_embeddings = torch.mul(omics_dataset,
                                                  omics_network_nodes_embedding_avg.expand(omics_dataset.shape[1],
                                                                                           omics_dataset.shape[0]).t())
        predictions = self.predictor(omics_dataset_with_embeddings)

        return predictions, omics_dataset_with_embeddings


def train_n(config, omics_dataset, omics_network_tg):
    X = omics_dataset.drop(['finalgold_visit'], axis=1)
    Y = omics_dataset['finalgold_visit']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # TODO: Double Check if the Dataset is already Standardized
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train.to_numpy())
    # X_test_scaled = scaler.transform(X_test.to_numpy())

    # Convert to PyTorch Tensors
    X_train_tensor = torch.FloatTensor(X_train.to_numpy())
    y_train_tensor = torch.LongTensor(y_train.to_numpy())

    X_test_tensor = torch.FloatTensor(X_test.to_numpy())
    y_test_tensor = torch.LongTensor(y_test.to_numpy())

    gnn_input_dim = omics_network_tg.x.shape[1]
    nn_input_dim = X_train_tensor.shape[1]
    nn_output_dim = 6  # Number of Classes # TODO: If we Merge Classes; then Update the Number of Classes here

    # lr_model = LogisticRegression(nn_input_dim, nn_output_dim)
    nn_model = NeuralNetwork(gnn_input_dim=gnn_input_dim,
                             gnn_hidden_dim=config['gnn_hidden_dim'],
                              gnn_layer_num=config['gnn_layer_num'],
                              ae_encoding_dim=1,
                              nn_input_dim=nn_input_dim, nn_hidden_dim1=config['nn_hidden_dim1'],
                              nn_hidden_dim2=config['nn_hidden_dim2'],
                              nn_output_dim=nn_output_dim)

    # Define the Loss Function and the Optimizer for the NN Model
    nn_criterion = nn.CrossEntropyLoss()
    nn_optimizer = optim.Adam(nn_model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # Define the Loss Function and the Optimizer for the LR Model
    # lr_criterion = nn.CrossEntropyLoss()
    # lr_optimizer = optim.Adam(lr_model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # Training loop
    for epoch in range(config['num_epochs']):
        nn_optimizer.zero_grad()
        # lr_optimizer.zero_grad()
        # Forward Pass
        output, omics_dataset_with_embeddings = nn_model(X_train_tensor, omics_network_tg)
        # lr_output = lr_model(X_train_tensor)
        # Compute the Loss Value
        loss = nn_criterion(output, y_train_tensor)
        # lr_loss = lr_criterion(lr_output, y_train_tensor)

        # Backward Propagation
        loss.backward()
        nn_optimizer.step()

        # lr_loss.backward()
        # lr_optimizer.step()

        # Print the Loss Value for Monitoring
        # print(f"Epoch [{epoch + 1}/{epoch}], Loss: {loss.item()}")

        if HYPERPARAMETERS_TUNING:
            metrics = {"loss": loss.item()}
            with tempfile.TemporaryDirectory() as tempdir:
                torch.save(
                    {"epoch": epoch, "model_state": nn_model.state_dict()},
                    os.path.join(tempdir, "checkpoint.pt"),
                )
                train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir))

        with torch.no_grad():
            nn_model.eval()
            # lr_model.eval()

            # lr_preds = lr_model(X_test_tensor)
            # l, predicted = torch.max(lr_preds, 1)
            # accuracy = torch.sum(predicted == y_test_tensor).item() / len(y_test)
            # print("Accuracy with LR: {:.4f}".format(accuracy))

            preds, omics_dataset_with_embeddings = nn_model(X_test_tensor, omics_network_tg)
            l, predicted = torch.max(preds, 1)
            accuracy = torch.sum(predicted == y_test_tensor).item() / len(y_test)

            # confusion_mtrx = confusion_matrix(y_test_tensor, predicted)
            # mc_auroc = MulticlassAUROC(num_classes=3, average='macro', thresholds=None)
            # print(f'Test Accuracy: {accuracy}')
            # print(f'Classification Report: {classification_report(y_test_tensor, predicted)}')
            # print(f'ROC AUC Score: {mc_auroc(outputs, y_test_tensor)}')
            # print(f'Confusion Matrix: {confusion_mtrx}')

            if HYPERPARAMETERS_TUNING:
                metrics["accuracy"] = accuracy
                with tempfile.TemporaryDirectory() as tempdir:
                    train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir))

    return accuracy


def main():
    args = make_args()

    # Setting up GPU Vs. CPU Usage
    if args.gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    else:
        print('Using CPU')

    # Setting up Hyperparameter Tuning
    if args.tune:
        print("Tuning the Hyperparameters of the Pipeline")
        global HYPERPARAMETERS_TUNING
        HYPERPARAMETERS_TUNING = True


    accuracies = []
    omics_datasets, omics_networks_tg = get_dataset()
    # print(omics_datasets)
    # print(omics_networks_tg)

    if args.tune:
        os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'
        for (omics_dataset, omics_network_tg) in zip(omics_datasets, omics_networks_tg):
            reporter = CLIReporter(metric_columns=["loss"])
            # Scheduler to Stop Bad Performing Trials
            scheduler = ASHAScheduler(
                metric="loss",  # TODO: Consider using Accuracy as a Metric
                mode="min",
                grace_period=10,
                reduction_factor=2)

            NUM_SAMPLES = 10000  # TODO: Need to Run without Limiting the Number of Samples
            result = tune.run(partial(train_n, omics_dataset=omics_dataset, omics_network_tg=omics_network_tg),
                              resources_per_trial={"cpu": 8, "gpu": 0},
                              config=PIPELINE_CONFIGS, num_samples=NUM_SAMPLES, scheduler=scheduler,
                              storage_path='/home/shussein/NetCO/GitHubCode/BioInformedSubjectRepresentation/BioInformedPipeline/raytune_results',
                              name='TuningFormerSmokers_%s' % omics_network_tg.x.shape[0],  # TODO: Update based on the Name of the Network
                              keep_checkpoints_num=1,
                              checkpoint_score_attr='min-loss',
                              progress_reporter=reporter
                              )
            best_trial = result.get_best_trial("loss", "min", "last")
            print("Best trial config: {}".format(best_trial.config))
            print("Best trial final test loss: {}".format(best_trial.last_result["loss"]))
    else:
        for (omics_dataset, omics_network_tg) in zip(omics_datasets, omics_networks_tg):
            print(omics_network_tg)
            for i in range(args.repeat_num):
                accuracy = train_n(config={
                    'gnn_layer_num': 4,
                    'gnn_hidden_dim': 16,
                    'lr': 0.029352058542109778,
                    'weight_decay': 0.00010178240820048331,
                    'nn_hidden_dim1': 8,
                    'nn_hidden_dim2': 4,
                    'num_epochs': 10,
                }, omics_dataset=omics_dataset, omics_network_tg=omics_network_tg)
                # print("Accuracy: {}".format(accuracy))
                accuracies.append(accuracy)
            print("Best Accuracy: {}".format(max(accuracies)))
            print("Average Accuracy is {}".format(sum(accuracies) / len(accuracies)))
            print("Standard Deviation for Avg Accuracy is {}".format(statistics.stdev(accuracies)))


if __name__ == "__main__":
    main()
