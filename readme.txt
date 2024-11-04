This is our implimentation for DTI-LEKG 

## Install and Usage
python version: 3.11.9

package require: torch pyg numpy scipy networkx pandas tqdm

run:

python src/main_args.py  --dataset yamanishi08_1_10  --split 1 --kg_dim 400 --gpu_inference cuda:0 --gpu_training cuda:0  --kg_training cuda:0  --kg_epoch 100

The two benchmark datasets can be accessed at https://zenodo.org/records/5500305

If you have any questions, please send a Email to hanzhang89@qq.com

details will be added soon
