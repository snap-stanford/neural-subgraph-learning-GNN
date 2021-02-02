# mlp
CUDA_VISIBLE_DEVICES=1 python3 -m subgraph_matching.alignment_expt --dataset=cox2 --model_path=ckpt/model-cox2-basis-curric-mlp.pt --use_whole_targets --method_type=mlp &
CUDA_VISIBLE_DEVICES=1 python3 -m subgraph_matching.alignment_expt --dataset=dd --model_path=ckpt/model-dd-basis-curric-mlp.pt --method_type=mlp &
CUDA_VISIBLE_DEVICES=1 python3 -m subgraph_matching.alignment_expt --dataset=msrc --model_path=ckpt/model-msrc-basis-curric-mlp.pt --use_whole_targets --method_type=mlp &
CUDA_VISIBLE_DEVICES=4 python3 -m subgraph_matching.alignment_expt --dataset=mmdb --model_path=ckpt/model-mmdb-basis-curric-mlp.pt --method_type=mlp &
CUDA_VISIBLE_DEVICES=4 python3 -m subgraph_matching.alignment_expt --dataset=enzymes --model_path=ckpt/model-enzymes-basis-curric-mlp.pt --use_whole_targets --method_type=mlp &
CUDA_VISIBLE_DEVICES=4 python3 -m subgraph_matching.alignment_expt --dataset=syn --model_path=ckpt/model-combsyn-mlp.pt --method_type=mlp &
