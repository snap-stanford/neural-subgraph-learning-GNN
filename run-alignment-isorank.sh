# isorank
CUDA_VISIBLE_DEVICES=0 python3 -m subgraph_matching.alignment_expt --dataset=cox2 --use_whole_targets --baseline=isorank &
CUDA_VISIBLE_DEVICES=0 python3 -m subgraph_matching.alignment_expt --dataset=dd --baseline=isorank &
CUDA_VISIBLE_DEVICES=0 python3 -m subgraph_matching.alignment_expt --dataset=msrc --use_whole_targets --baseline=isorank &
CUDA_VISIBLE_DEVICES=2 python3 -m subgraph_matching.alignment_expt --dataset=mmdb --baseline=isorank &
CUDA_VISIBLE_DEVICES=2 python3 -m subgraph_matching.alignment_expt --dataset=enzymes --use_whole_targets --baseline=isorank &
CUDA_VISIBLE_DEVICES=2 python3 -m subgraph_matching.alignment_expt --dataset=syn --baseline=isorank &

