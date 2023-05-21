# Equations: 
# eval_interval = num_graph * subgraph_per_graph * 4 / batch_size
# n_batches = eval_interval * epochs

# Real Datasets:

python -m subgraph_matching.train --dataset preloaded-../giso/data_processed/KKI --batch_size 64 --eval_interval 10375 --n_batches 311250 --model_path="ckpt/KKI.pt"
python -m subgraph_matching.train --dataset preloaded-../giso/data_processed/DBLP-v1 --batch_size 64 --eval_interval 121600 --n_batches 3648000 --model_path="ckpt/DBLP-v1.pt"
python -m subgraph_matching.train --dataset preloaded-../giso/data_processed/COX2 --batch_size 64 --eval_interval 58375 --n_batches 1751250 --model_path="ckpt/COX2.pt"
python -m subgraph_matching.train --dataset preloaded-../giso/data_processed/MSRC-21 --batch_size 64 --eval_interval 70375 --n_batches 2111250 --model_path="ckpt/MSRC-21.pt"
python -m subgraph_matching.train --dataset preloaded-../giso/data_processed/DHFR --batch_size 64 --eval_interval 94500 --n_batches 2835000 --model_path="ckpt/DHFR.pt"


python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/KKI --model_path="ckpt/KKI.pt"
python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/KKI-dense_0_20 --model_path="ckpt/KKI.pt"
python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/KKI-dense_20_40 --model_path="ckpt/KKI.pt"
python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/KKI-dense_40_60 --model_path="ckpt/KKI.pt"
python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/KKI-dense_60_ --model_path="ckpt/KKI.pt"
python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/KKI-nondense_0_20 --model_path="ckpt/KKI.pt"
python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/KKI-nondense_20_40 --model_path="ckpt/KKI.pt"
python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/KKI-nondense_40_60 --model_path="ckpt/KKI.pt"
python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/KKI-nondense_60_ --model_path="ckpt/KKI.pt"


python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/DBLP-v1 --model_path="ckpt/DBLP-v1.pt"
python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/DBLP-v1-dense_0_20 --model_path="ckpt/DBLP-v1.pt"
python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/DBLP-v1-dense_20_40 --model_path="ckpt/DBLP-v1.pt"
python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/DBLP-v1-nondense_0_20 --model_path="ckpt/DBLP-v1.pt"
python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/DBLP-v1-nondense_20_40 --model_path="ckpt/DBLP-v1.pt"


python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/COX2 --model_path="ckpt/COX2.pt"
python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/COX2-dense_0_20 --model_path="ckpt/COX2.pt"
python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/COX2-dense_20_40 --model_path="ckpt/COX2.pt"
python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/COX2-dense_40_60 --model_path="ckpt/COX2.pt"
python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/COX2-dense_60_ --model_path="ckpt/COX2.pt"
python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/COX2-nondense_0_20 --model_path="ckpt/COX2.pt"
python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/COX2-nondense_20_40 --model_path="ckpt/COX2.pt"
python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/COX2-nondense_40_60 --model_path="ckpt/COX2.pt"
python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/COX2-nondense_60_ --model_path="ckpt/COX2.pt"


python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/MSRC-21 --model_path="ckpt/MSRC-21.pt"
python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/MSRC-21-dense_0_20 --model_path="ckpt/MSRC-21.pt"
python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/MSRC-21-dense_20_40 --model_path="ckpt/MSRC-21.pt"
python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/MSRC-21-dense_40_60 --model_path="ckpt/MSRC-21.pt"
python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/MSRC-21-dense_60_ --model_path="ckpt/MSRC-21.pt"
python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/MSRC-21-nondense_0_20 --model_path="ckpt/MSRC-21.pt"
python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/MSRC-21-nondense_20_40 --model_path="ckpt/MSRC-21.pt"
python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/MSRC-21-nondense_40_60 --model_path="ckpt/MSRC-21.pt"
python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/MSRC-21-nondense_60_ --model_path="ckpt/MSRC-21.pt"


python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/DHFR --model_path="ckpt/DHFR.pt"
python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/DHFR-nondense_0_20 --model_path="ckpt/DHFR.pt"
python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/DHFR-nondense_20_40 --model_path="ckpt/DHFR.pt"
python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/DHFR-nondense_40_60 --model_path="ckpt/DHFR.pt"
python -m subgraph_matching.test --dataset preloaded-../giso/data_processed/DHFR-nondense_60_ --model_path="ckpt/DHFR.pt"



# Synthesis Datasets:

python -m subgraph_matching.train --dataset preloaded-../giso/data_processed/large_40_4_20 --batch_size 64 --eval_interval 500000 --n_batches 5000000 --model_path="ckpt/large_40_4_20.pt"
python -m subgraph_matching.train --dataset preloaded-../giso/data_processed/large_60_4_20 --batch_size 64 --eval_interval 500000 --n_batches 5000000 --model_path="ckpt/large_60_4_20.pt"
python -m subgraph_matching.train --dataset preloaded-../giso/data_processed/large_80_4_20 --batch_size 64 --eval_interval 500000 --n_batches 5000000 --model_path="ckpt/large_80_4_20.pt"
python -m subgraph_matching.train --dataset preloaded-../giso/data_processed/large_100_4_20 --batch_size 64 --eval_interval 500000 --n_batches 5000000 --model_path="ckpt/large_100_4_20.pt"
