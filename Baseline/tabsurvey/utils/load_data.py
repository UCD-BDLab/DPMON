import pandas as pd

def load_data(args):
    # ex, args.dataset = 'CurrentSmokersNetwork1-98'
    path = ""
    value = args.dataset[:-4]
    network = args.dataset
    print(value)
    print(network)

    if value == "CurrentSmokersNetwork":
        path = "../../BioInformedSubjectRepresentation/Current/"
        omics_data = "current_smokers_omics_data"
        gold = "current_smokers_gold"
    elif value == "FormerSmokersNetwork":
        path = "../../BioInformedSubjectRepresentation/Former/"
        omics_data = "former_smokers_omics_data"
        gold = "former_smokers_gold"

    selected_network = pd.read_csv(f"{path}{network}.csv")
    list_columns = selected_network.columns.tolist()
    list_columns.remove('Unnamed: 0')

    smokers_omics = pd.read_csv(f"{path}{omics_data}.csv")
    data = smokers_omics[list_columns]

    X = data.values

    # Target class stored in the gold csv file which is a single column
    target_class = pd.read_csv(f"{path}{gold}.csv")

    # Change the value of the target class -1 to 5
    for index, row in target_class.iterrows():
        if row["finalgold_visit"] == -1: 
            target_class.loc[index, "finalgold_visit"] = 5

    y = target_class['finalgold_visit'].values
    print(set(y))

    return X, y