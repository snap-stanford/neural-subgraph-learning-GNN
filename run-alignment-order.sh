# order emb
CUDA_VISIBLE_DEVICES=1 python3 -m subgraph_matching.alignment_expt --dataset=cox2 --model_path=ckpt/model-cox2-basis-curric.pt --use_whole_targets &
CUDA_VISIBLE_DEVICES=1 python3 -m subgraph_matching.alignment_expt --dataset=dd --model_path=ckpt/model-dd-basis-curric.pt &
CUDA_VISIBLE_DEVICES=1 python3 -m subgraph_matching.alignment_expt --dataset=msrc --model_path=ckpt/model-msrc-basis-curric.pt --use_whole_targets &
CUDA_VISIBLE_DEVICES=3 python3 -m subgraph_matching.alignment_expt --dataset=mmdb --model_path=ckpt/model-mmdb-basis-curric.pt &
CUDA_VISIBLE_DEVICES=3 python3 -m subgraph_matching.alignment_expt --dataset=enzymes --model_path=ckpt/model-enzymes-basis-curric.pt --use_whole_targets &
CUDA_VISIBLE_DEVICES=3 python3 -m subgraph_matching.alignment_expt --dataset=syn --model_path=ckpt/model-combsyn.pt &
