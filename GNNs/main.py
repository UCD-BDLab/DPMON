# TODO: Refactor the code - Remove unnecessary print statements
# TODO: Add support in the code for the two modes (regular and hyperparameter tuning)
# TODO: Finalize the list of parameters to fine-tune

import re
import os
from functools import partial
from torch_geometric.explain import Explainer, GNNExplainer
from sklearn.model_selection import ParameterGrid
from tensorboardX import SummaryWriter
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.loader import DataLoader

from args import *
from model import *
from utils import *
from dataset import *

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# TODO: Create a Different Config for Each Flavor of GNN
GCN_CONFIG = {
    "lr": tune.loguniform(1e-4, 1e-1),
    "weight_decay": tune.loguniform(1e-4, 1e-1),
    "layer_num": tune.choice([2, 4, 8, 16, 32, 64, 128]),
    "hidden_dim": tune.choice([4, 8, 16, 32, 64, 128]),
    "epoch_num": tune.choice([2, 16, 64, 512, 1024, 4096, 8192]),
}


def train(config, args, datasets_name):
    device = torch.device("cuda:" + str(args.cuda) if args.gpu else "cpu")

    for dataset_name in datasets_name:
        # TODO: Double Check that the 'runs' Directory is the Correct Directory for this?
        writer_train = SummaryWriter(comment=dataset_name + "_" + args.model + "_train")
        writer_val = SummaryWriter(comment=dataset_name + "_" + args.model + "_val")
        writer_test = SummaryWriter(comment=dataset_name + "_" + args.model + "_test")

        results = []
        for repeat in range(args.repeat_num):
            result_val = []
            result_test = []
            start_time = time.time()

            dataset = get_tg_dataset(args, dataset_name, use_cache=args.cache)
            dataset_load_time = time.time() - start_time
            print(dataset_name, "load time", dataset_load_time)

            print("Dataset %s" % dataset)

            # Understading how they select the anchor_set
            # preselect_anchor(data, layer_num=config['layer_num'], anchor_num=config['anchor_num'], device='cpu')

            transform = RandomNodeSplit(num_val=3, num_test=5)
            data = transform(dataset)
            dataloader = DataLoader([data], batch_size=1, shuffle=True)
            # Figuring out how to send data to GPU with dataloader
            # data = data.to(device)

            # Model
            num_features = dataset.x.shape[1]
            input_dim = num_features
            edge_weight_dim = dataset.edge_attr.shape[0]
            output_dim = 1

            # TODO: Figure out a better way to make this dynamic (right now it only runs for PGNN)

            # model = PGNN(input_dim=input_dim, feature_dim=args.feature_dim,
            #              hidden_dim=config['hidden_dim'], output_dim=output_dim,
            #              feature_pre=args.feature_pre, layer_num=config['layer_num'],
            #              dropout=args.dropout, hidden_dim_name=dataset_name).to(device)

            model = GCN(
                input_dim=input_dim,
                hidden_dim=config["hidden_dim"],
                output_dim=output_dim,
                layer_num=config["layer_num"],
                dropout=args.dropout,
                hidden_dim_name=dataset_name,
            ).to(device)

            # model = GCNExplainer(input_dim=input_dim, hidden_dim=config['hidden_dim'], output_dim=output_dim,
            #             layer_num=config['layer_num'], dropout=args.dropout, hidden_dim_name=dataset_name).to(device)

            # model = GCN2(input_dim=input_dim, hidden_dim=config['hidden_dim'], output_dim=output_dim,
            #              hidden_dim_name=dataset_name, layer_num=config['layer_num'], alpha=0.1, theta=0.5,
            #              shared_weights=True, dropout=args.dropout).to(device)

            # model = GAT(input_dim=input_dim, hidden_dim=config['hidden_dim'], output_dim=output_dim,
            #             layer_num=config['layer_num'], dropout=args.dropout, hidden_dim_name=dataset_name).to(device)

            # model = GAT2(input_dim=input_dim, hidden_dim=config['hidden_dim'], output_dim=output_dim,
            #             layer_num=config['layer_num'], dropout=args.dropout, hidden_dim_name=dataset_name).to(device)

            # model = SAGE(input_dim=input_dim, hidden_dim=config['hidden_dim'], output_dim=output_dim,
            #             layer_num=config['layer_num'], dropout=args.dropout, hidden_dim_name=dataset_name).to(device)

            # model = GIN(input_dim=input_dim, hidden_dim=config['hidden_dim'], output_dim=output_dim,
            #              layer_num=config['layer_num'], dropout=args.dropout, hidden_dim_name=dataset_name).to(device)

            # Loss
            # optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=5e-4)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
            )
            # optimizer = torch.optim.Adam([
            #     dict(params=model.convs.parameters(), weight_decay=0.01),
            #     dict(params=model.linear.parameters(), weight_decay=5e-4)
            # ], lr=0.01)
            loss_func = torch.nn.MSELoss()

            for epoch in range(config["epoch_num"]):
                print("********** Epoch %d" % epoch)
                loss_train = 0
                if epoch == 200:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] /= 10

                # if args.permute:
                #     preselect_anchor(data, layer_num=config['layer_num'], anchor_num=config['anchor_num'], device=device)

                for batch in dataloader:
                    # print("Batch %s" % batch.train_mask)
                    # print("Data %s" % data)
                    print("Before Train")
                    model.train()
                    print("After Train")
                    optimizer.zero_grad()
                    if args.model == "GCNExplainer":
                        out = model(batch.x, batch.edge_index).flatten()
                    else:
                        print("Before Out")
                        out = model(batch).flatten()
                        print("After Out")
                    # print("Expected %s\nPredicted %s" % (batch.y[batch.train_mask], out[batch.train_mask]))
                    loss = loss_func(out[batch.train_mask], batch.y[batch.train_mask])

                    # update
                    loss.backward()
                    optimizer.step()
                    # print("Actual %s\nPredicted: %s" % (batch.y[batch.train_mask], out[batch.train_mask]))

                    loss_train += (
                        loss_func(out[batch.train_mask], batch.y[batch.train_mask])
                        .cpu()
                        .data.numpy()
                    )

                loss_train /= len(dataloader.dataset)

                if epoch % args.epoch_log == 0:
                    # evaluate
                    print("Before Eval")
                    model.eval()
                    print("After Eval")
                    loss_train = 0
                    loss_val = 0
                    loss_test = 0

                    if args.model == "GCNExplainer":
                        out = model(batch.x, batch.edge_index).flatten()
                    else:
                        print("Before Out Eval")
                        out = model(batch).flatten()
                        print("After Out Eval")
                    print(
                        "[Train] Expected: %s\nActual: %s"
                        % (batch.y[batch.train_mask], out[batch.train_mask])
                    )
                    loss_train += (
                        loss_func(out[batch.train_mask], batch.y[batch.train_mask])
                        .cpu()
                        .data.numpy()
                    )
                    loss_val += (
                        loss_func(out[batch.val_mask], batch.y[batch.val_mask])
                        .cpu()
                        .data.numpy()
                    )
                    loss_test += (
                        loss_func(out[batch.test_mask], batch.y[batch.test_mask])
                        .cpu()
                        .data.numpy()
                    )

                    print(
                        repeat,
                        epoch,
                        "Train Loss {:.4f}".format(loss_train),
                        "Validate Loss {:.4f}".format(loss_val),
                        "Test Loss {:.4f}".format(loss_test),
                    )

                    writer_train.add_scalar(
                        "repeat_" + str(repeat) + "/loss_" + dataset_name,
                        loss_train,
                        epoch,
                    )
                    writer_val.add_scalar(
                        "repeat_" + str(repeat) + "/loss_" + dataset_name,
                        loss_val,
                        epoch,
                    )
                    writer_test.add_scalar(
                        "repeat_" + str(repeat) + "/loss_" + dataset_name,
                        loss_test,
                        epoch,
                    )

                    result_val.append(loss_val)
                    result_test.append(loss_test)
                    if args.tune:
                        with tune.checkpoint_dir(epoch) as checkpoint_dir:
                            path = os.path.join(checkpoint_dir, "checkpoint")
                            torch.save(
                                (model.state_dict(), optimizer.state_dict()), path
                            )
                        tune.report(loss=loss_val)

            result_val = np.array(result_val)
            result_test = np.array(result_test)
            results.append(result_val[np.argmin(result_val)])
        print("-----------------Final-------------------")
        print(results)

        # ######################### GNN Explainer ####################33
        # explainer = Explainer(
        #     model=model,
        #     algorithm=GNNExplainer(epochs=500),
        #     explanation_type='model',
        #     node_mask_type='attributes',
        #     edge_mask_type='object',
        #     model_config=dict(
        #         # mode='multiclass_classification',
        #         mode='regression',
        #         task_level='node',
        #         return_type='raw',
        #     ),
        # )
        # for node_index in range(data.x.shape[0]):
        #     explanation = explainer(data.x, data.edge_index, index=node_index)
        #     print(f'Generated explanations in {explanation.available_explanations}')
        #
        #     path = 'feature_importance_%s.png' % node_index
        #     explanation.visualize_feature_importance(path, top_k=10)
        #     print(f"Feature importance plot has been saved to '{path}'")
        #
        #     path = 'subgraph_%s.pdf' % node_index
        #     explanation.visualize_graph(path)
        #     print(f"Subgraph visualization plot has been saved to '{path}'")


def main():
    if not os.path.isdir("results"):
        os.mkdir("results")

    args = make_args()

    # Setting up GPU Vs. CPU Usage
    if args.gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
        print("Using GPU {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    else:
        print("Using CPU")

    if args.dataset == "all":
        datasets_name = []
        for _, _, files in os.walk(DATASET_DIR):
            for file in files:
                if re.compile("trimmed_fev1_.*_.*_adj\.csv").match(file):
                    datasets_name.append(file)
    else:
        datasets_name = [args.dataset]

    # Hyperparameter Tuning
    if args.tune:
        for dataset_name in datasets_name:
            reporter = CLIReporter(metric_columns=["loss"])
            GRACE_PERIOD = 10
            # Scheduler to stop bad performing trials
            scheduler = ASHAScheduler(
                metric="loss", mode="min", grace_period=GRACE_PERIOD, reduction_factor=2
            )

            # Limit on the number of trials
            # NUM_SAMPLES = 5000
            NUM_SAMPLES = 1000
            result = tune.run(
                partial(train, args=args, datasets_name=[dataset_name]),
                resources_per_trial={"cpu": 8, "gpu": 0},
                config=GCN_CONFIG,
                num_samples=NUM_SAMPLES,
                scheduler=scheduler,
                local_dir="outputs/raytune_results_%s" % dataset_name,
                name="GCN_SHAP",
                keep_checkpoints_num=1,
                checkpoint_score_attr="min-loss",
                progress_reporter=reporter,
            )
            best_trial = result.get_best_trial("loss", "min", "last")
            print("Best trial config: {}".format(best_trial.config))
            print(
                "Best trial final test loss: {}".format(best_trial.last_result["loss"])
            )
    else:
        train(
            config={
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "layer_num": args.layer_num,
                "hidden_dim": args.hidden_dim,
                "epoch_num": args.epoch_num,
            },
            args=args,
            datasets_name=datasets_name,
        )


if __name__ == "__main__":
    main()
