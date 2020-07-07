def parse_encoder(parser):
    parser.opt_list('--conv_type', type=str, tunable=True, 
            options=['GIN', 'SAGE'],#, 'GCN'],#, 'GAT'],
            help='type of model')
    parser.opt_list('--skip', type=str, tunable=True, 
            options=['all', 'last'],#, 'GCN'],#, 'GAT'],
            help='type of model')
    parser.opt_list('--method_type', type=str, tunable=True,
            options=['order'],
            help='type of convolution') # can change name to embedding_type
    parser.opt_list('--order_func_grid_size', type=int, tunable=False,
            options=[1000])
    parser.opt_list('--batch_size', type=int, tunable=False,
            help='Training batch size')
    parser.opt_list('--n_layers', type=int, tunable=True, 
            options=[4, 8, 12],
            help='Number of graph conv layers')
    parser.opt_list('--hidden_dim', type=int, tunable=False,
            help='Training hidden size')
    parser.opt_list('--dropout', type=float, tunable=False,
            help='Dropout rate')
    parser.opt_list('--margin', type=float, tunable=False,
            help='margin for loss')
    parser.opt_list('--regularization', type=float, tunable=False,
            help='regularization coeff')

    # non-tunable
    parser.add_argument('--n_inner_layers', type=int,
                        help='Number of inner graph conv layers (gatedgraphconv)')
    parser.add_argument('--max_graph_size', type=int,
                        help='max training graph size')
    parser.add_argument('--n_batches', type=int,
                        help='Number of training minibatches')
    parser.add_argument('--dataset', type=str,
                        help='Dataset')
    parser.add_argument('--dataset_type', type=str,
                        help='"syn" or "real"')
    parser.add_argument('--test_set', type=str,
                        help='test set filename')
    parser.add_argument('--eval_interval', type=int,
                        help='how often to eval during training')
    parser.add_argument('--val_size', type=int,
                        help='validation set size')
    parser.add_argument('--model_path', type=str,
                        help='path to save/load model')
    parser.add_argument('--start_weights', type=str,
                        help='file to load weights from')
    parser.add_argument('--opt_scheduler', type=str,
                        help='scheduler name')
    parser.add_argument('--use_intersection', type=bool,
                        help='whether to use intersections in training')
    parser.add_argument('--use_diverse_motifs', action="store_true",
                        help='whether to use diverse motifs in training')
    parser.add_argument('--node_anchored', action="store_true",
                        help='whether to use node anchoring in training')
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--n_workers', type=int)
    parser.add_argument('--tag', type=str,
        help='tag to identify the run')

    parser.set_defaults(conv_type='SAGE',
                        method_type='order',
                        dataset='syn',
                        n_layers=8,
                        batch_size=64,
                        hidden_dim=64,
                        skip="learnable",
                        dropout=0.0,
                        n_batches=1000000,
                        opt='adam',   # opt_enc_parser
                        opt_scheduler='none',
                        opt_restart=100,
                        weight_decay=0.0,
                        lr=1e-4,
                        margin=0.1,
                        test_set='',
                        eval_interval=1000,
                        n_workers=4,
                        model_path="ckpt/model.pt",
                        tag='',
                        val_size=4096,
                        node_anchored=True)
