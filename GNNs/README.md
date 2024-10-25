## Setting up Virtual Enviroment.

On windows you may need to install: pip install pyenv-win==1.2.1

This is because ray is only availble on python 3.9 and 3.10

To run the code, use the following command: python main.py --model <NameoftheModel> --dataset <NameoftheDataset> --lr <LearningRate> --weight-decay <WeightDecay> --layer_num <NumberofLayers> --hidden_dim <EmbeddingDim> --epoch_num <NumberofEpochs>; where:

    --model: The GNN flavor that you want to run (GCN, GAT, SAGE, ..etc.).
    --dataset: The name of the dataset that you want to run with (this is the graph dataset in the form of a CSV file of the adjacency matrix).
    --lr: The learning rate for the GNN (how often the model updates its parameters. ()
    --weight_decay: The weight decay for the GNN model.
    --layer_num: Number of layers for the GNN model. How many neighbors for each node are we collecting information from.
    --hidden_dim: The dimension of the embedding space.
    --epoch_num: The number of iterations for the training process.
