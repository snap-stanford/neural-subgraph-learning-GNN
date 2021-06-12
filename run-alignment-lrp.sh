# lrp
CUDA_VISIBLE_DEVICES=1 python3 -m subgraph_matching.alignment_expt --dataset=cox2 --model_path=ckpt/model-cox2-basis-curric-lrp.pt --use_whole_targets --method_type=lrp &
CUDA_VISIBLE_DEVICES=1 python3 -m subgraph_matching.alignment_expt --dataset=dd --model_path=ckpt/model-dd-basis-curric-lrp.pt --method_type=lrp &
CUDA_VISIBLE_DEVICES=1 python3 -m subgraph_matching.alignment_expt --dataset=msrc --model_path=ckpt/model-msrc-basis-curric-lrp.pt --use_whole_targets --method_type=lrp &
CUDA_VISIBLE_DEVICES=4 python3 -m subgraph_matching.alignment_expt --dataset=mmdb --model_path=ckpt/model-mmdb-basis-curric-lrp.pt --method_type=lrp &
CUDA_VISIBLE_DEVICES=4 python3 -m subgraph_matching.alignment_expt --dataset=enzymes --model_path=ckpt/model-enzymes-basis-curric-lrp.pt --use_whole_targets --method_type=lrp &
CUDA_VISIBLE_DEVICES=4 python3 -m subgraph_matching.alignment_expt --dataset=syn --model_path=ckpt/model-combsyn-lrp.pt --method_type=lrp &

