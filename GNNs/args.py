from argparse import ArgumentParser

def make_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        dest="model",
        default="GCN",
        type=str,
        help="model class name. E.g., GCN, PGNN, ...",
    )
    parser.add_argument(
        "--dataset", dest="dataset", default="All", type=str, help="All"
    )
    parser.add_argument(
        "--gpu", dest="gpu", action="store_true", help="whether use gpu"
    )
    parser.add_argument(
        "--tune",
        dest="tune",
        action="store_true",
        help="whether to tune the model hyperparameters",
    )
    parser.add_argument(
        "--cache_no", dest="cache", action="store_false", help="whether use cache"
    )
    parser.add_argument(
        "--cpu", dest="gpu", action="store_false", help="whether use cpu"
    )
    parser.add_argument("--cuda", dest="cuda", default="0", type=str)

    # Dataset Notes: Enable after Investigation
    # parser.add_argument('--feature_pre', dest='feature_pre', action='store_true',
    #                     help='whether pre transform feature')
    # parser.add_argument('--feature_pre_no', dest='feature_pre', action='store_false',
    #                     help='whether pre transform feature')

    parser.add_argument(
        "--dropout",
        dest="dropout",
        default=0.5,
        type=float,
        help="Dropout Value, Default is 0.5",
    )

    parser.add_argument("--layer_num", dest="layer_num", default=2, type=int)
    parser.add_argument("--hidden_dim", dest="hidden_dim", default=32, type=int)
    parser.add_argument("--output_dim", dest="output_dim", default=32, type=int)
    parser.add_argument("--anchor_num", dest="anchor_num", default=64, type=int)

    parser.add_argument("--lr", dest="lr", default=1e-2, type=float)
    parser.add_argument("--weight_decay", dest="weight_decay", default=1e-2, type=float)
    parser.add_argument("--epoch_num", dest="epoch_num", default=2001, type=int)
    parser.add_argument("--repeat_num", dest="repeat_num", default=1, type=int)
    parser.add_argument("--epoch_log", dest="epoch_log", default=10, type=int)

    parser.set_defaults(
        gpu=True,
        model="GCN",
        dataset="All",
        cache=False,
        permute=True,
        dropout=0.5,
        approximate=-1,
    )
    args = parser.parse_args()
    return args
