# Subspace Adaptation Prior for Few-Shot Learning

This is the Github repository accompanying the paper titled *Subspace Adaptation Prior for Few-Shot Learning*. Here, you find the code that we have used for our experiments and instructions on how to reproduce the results. 


## Step 1: Fetching the datasets

In order to run the code, download the following datasets. 
- miniImagenet: https://drive.google.com/uc?export=download&id=16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY
- tieredImagenet: https://drive.google.com/uc?export=download&id=1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07
- CUB: https://drive.google.com/uc?export=download&id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45

After downloading the datasets, place the downloaded files in the following location. 
- miniImagenet: ./data/miniimagenet/mini-imagenet.tar.gz
- tieredImagenet: ./data/tieredimagenet/tiered-imagenet.tar
- CUB: ./data/cub/CUB_200_2011.tgz

## Step 2: Reproducing the results

Below, you find the exact commands required to re-run the experimental results on few-shot sine wave regression and image classification. 

### Few-shot sine wave regression

Note that running the commands below require having a directory called *saplogs*. These commands are for when making 1 gradient update step per task. Replacing --T 1 by --T 10 will give you the results for 10 steps per task. 

**MAML**

nohup python -u main.py --problem sine --N 1 --k 5 --k_test 50 --model maml --model_spec fmaml --cpu --val_after 2500 --second_order --T 1 --runs 5 --validate --meta_batch_size 4 > saplogs/sine/sine-maml-k5.log &

nohup python -u main.py --problem sine --N 1 --k 10 --k_test 50 --model maml --model_spec fmaml --cpu --val_after 2500 --second_order --T 1 --runs 5 --validate --meta_batch_size 4 > saplogs/sine/sine-maml-k10.log &


**T-Net**

nohup python -u main.py --problem sine --N 1 --k 5 --k_test 50 --model sap --model_spec ftnet --cpu --val_after 2500 --second_order --T 1 --gamma 0 --runs 5 --reg null --validate --meta_batch_size 4 --tnet > saplogs/sine/sine-tnet-k5.log &
 
nohup python -u main.py --problem sine --N 1 --k 10 --k_test 50 --model sap --model_spec ftnet --cpu --val_after 2500 --second_order --T 1 --gamma 0 --runs 5 --reg null --validate --meta_batch_size 4 --tnet > saplogs/sine/sine-tnet-k10.log &


**MT-Net**

nohup python -u main.py --problem sine --N 1 --k 5 --k_test 50 --model sap --model_spec fmtnet --cpu --val_after 2500 --second_order --T 1 --gamma 0 --runs 5 --reg null --validate --meta_batch_size 4 --tnet --use_grad_mask > saplogs/sine/sine-mtnet-k5.log &
 
nohup python -u main.py --problem sine --N 1 --k 10 --k_test 50 --model sap --model_spec fmtnet --cpu --val_after 2500 --second_order --T 1 --gamma 0 --runs 5 --reg null --validate --meta_batch_size 4 --tnet --use_grad_mask > saplogs/sine/sine-mtnet-k10.log &


**SAP**

nohup python -u main.py --problem sine --N 1 --k 5 --k_test 50 --model sap --model_spec fsap-svd --cpu --val_after 2500 --second_order --T 1 --gamma 0 --runs 5 --learn_alfas --reg null --svd --validate --meta_batch_size 4 > saplogs/sine/sine-sap-svd-k5.log &
 
nohup python -u main.py --problem sine --N 1 --k 10 --k_test 50 --model sap --model_spec fsap-svd --cpu --val_after 2500 --second_order --T 1 --gamma 0 --runs 5 --learn_alfas --reg null --svd --validate --meta_batch_size 4 > saplogs/sine/sine-sap-svd-k10.log &


nohup python -u main.py --problem sine --N 1 --k 5 --k_test 50 --model sap --model_spec fmsap-svd --cpu --val_after 2500 --second_order --T 1 --gamma 0 --runs 5 --learn_alfas --reg null --svd --validate --meta_batch_size 4 --use_grad_mask > saplogs/sine/sine-msap-svd-k5.log &
 
nohup python -u main.py --problem sine --N 1 --k 10 --k_test 50 --model sap --model_spec fmsap-svd --cpu --val_after 2500 --second_order --T 1 --gamma 0 --runs 5 --learn_alfas --reg null --svd --validate --meta_batch_size 4 --use_grad_mask > saplogs/sine/sine-msap-svd-k10.log &


### Few-shot image classification

#### miniImageNet

**MAML**: 

python -u main.py --problem min --k_test 16  --backbone conv4 --model maml --validate --val_after 8000 --T 5 --k 1 --N 5 --T_test 10 --T_val 10 --meta_batch_size 4 --runs 1 --single_run --seed $1  --base_lr 0.01 --model_spec maml-reproduce-final --second_order --grad_clip 10 --out_channels 32  --train_iters 240000 

python -u main.py --problem min --k_test 16  --backbone conv4 --model maml --validate --val_after 4000 --T 5 --k 5 --N 5 --T_test 10 --T_val 10 --meta_batch_size 2 --runs 1 --single_run --seed $1  --base_lr 0.01 --model_spec maml-reproduce-final --second_order --grad_clip 10 --out_channels 32  --train_iters 120000 


python -u main.py --problem min --k_test 16  --backbone conv4 --model maml --validate --val_after 8000 --T 5 --k 1 --N 5 --T_test 10 --T_val 10 --meta_batch_size 4 --runs 1 --single_run --seed $1  --base_lr 0.01 --model_spec maml-reproduce-final-64c--second_order --grad_clip 10 --out_channels 64  --train_iters 240000 --cross_eval

python -u main.py --problem min --k_test 16  --backbone conv4 --model maml --validate --val_after 4000 --T 5 --k 5 --N 5 --T_test 10 --T_val 10 --meta_batch_size 2 --runs 1 --single_run --seed $1  --base_lr 0.01 --model_spec maml-reproduce-final-64c --second_order --grad_clip 10 --out_channels 64  --train_iters 120000 --cross_eval



**T-Net** (c=32,64): 

python -u main.py --problem min --k_test 16  --backbone conv4 --model sap --validate --val_after 8000 --T 5 --k 1 --N 5 --T_test 10 --T_val 10 --meta_batch_size 4 --runs 1 --single_run --seed $1 --base_lr 0.01 --model_spec tnet-reproduce-final-$2c --second_order --grad_clip 10 --out_channels $2 --tnet --gamma 0 --reg null  --train_iters 240000 --cross_eval

python -u main.py --problem min --k_test 16  --backbone conv4 --model sap --validate --val_after 4000 --T 5 --k 5 --N 5 --T_test 10 --T_val 10 --meta_batch_size 2 --runs 1 --single_run --seed $1 --base_lr 0.01 --model_spec tnet-reproduce-final-$2c --second_order --grad_clip 10 --out_channels $2 --tnet --gamma 0 --reg null  --train_iters 120000 --cross_eval



**MT-Net** (c=32,64):  

python -u main.py --problem min --k_test 16  --backbone conv4 --model sap --validate --val_after 8000 --T 5 --k 1 --N 5 --T_test 10 --T_val 10 --meta_batch_size 4 --runs 1 --single_run --seed $1 --base_lr 0.01 --model_spec mtnet-reproduce-final-$2c --second_order --grad_clip 10 --out_channels $2 --tnet --gamma 0 --reg null  --train_iters 240000 --use_grad_mask --cross_eval

python -u main.py --problem min --k_test 16  --backbone conv4 --model sap --validate --val_after 4000 --T 5 --k 5 --N 5 --T_test 10 --T_val 10 --meta_batch_size 2 --runs 1 --single_run --seed $1 --base_lr 0.01 --model_spec mtnet-reproduce-final-$2c --second_order --grad_clip 10 --out_channels $2 --tnet --gamma 0 --reg null  --train_iters 120000 --use_grad_mask --cross_eval



**WarpGrad**:

python -u main.py --problem min --k_test 16  --backbone conv4 --model sap --val_after 2500 --T 5 --k $1 --k_train 5 --N 5 --T_test 5 --T_val 5 --meta_batch_size 1 --runs 1 --single_run --base_lr 0.1 --model_spec warpgrad-reproduce-final-lr0.1-c64 --second_order --out_channels 64 --transform_out_channels 64 --tnet --gamma 0 --reg null --warpgrad --use_bias --validate --train_iters 60000  --seed $2 --cross_eval

python -u main.py --problem min --k_test 16  --backbone conv4 --model sap --val_after 2500 --T 5 --k $1 --k_train 5 --N 5 --T_test 5 --T_val 5 --meta_batch_size 1 --runs 1 --single_run --base_lr 0.1 --model_spec warpgrad-reproduce-final-lr0.1-c32 --second_order --out_channels 32 --transform_out_channels 32 --tnet --gamma 0 --reg null --warpgrad --use_bias --validate --train_iters 60000  --seed $2 --cross_eval

**SAP**:
python -u main.py --problem min --N 5 --k $2 --k_test 16 --model sap --model_spec fsap-best-T1-MBS4 --linear_transform --val_after 2500 --second_order --T 1 --gamma 0 --runs 1 --learn_alfas --reg null --T_test 10 --meta_batch_size 4 --T_val 10 --channel_scale --svd --grad_clip 10 --old --base_lr 0.0360774985854036 --seed $1 --single_run --validate --cross_eval  


python -u main.py --problem min --N 5 --k $2 --k_test 16 --model sap --model_spec fsap-best-T1-MBS4-32c --linear_transform --val_after 2500 --second_order --T 1 --gamma 0 --runs 1 --learn_alfas --reg null --T_test 10 --meta_batch_size 4 --T_val 10 --channel_scale --svd --grad_clip 10 --old --base_lr 0.0360774985854036 --seed $1 --single_run --validate --cross_eval --out_channels 32  

python -u main.py --problem min --N 5 --k $2 --k_test 16 --model sap --model_spec fsap-best-T1-MBS4-32c-FO --linear_transform --val_after 2500 --T 1 --gamma 0 --runs 1 --learn_alfas --reg null --T_test 10 --meta_batch_size 4 --T_val 10 --channel_scale --svd --grad_clip 10 --old --base_lr 0.0360774985854036 --seed $1 --single_run --validate --cross_eval --out_channels 32  


#### tieredImageNet

**MAML**: 

python -u main.py --problem tiered --k_test 16  --backbone conv4 --model maml --validate --val_after 8000 --T 5 --k 1 --N 5 --T_test 10 --T_val 10 --meta_batch_size 4 --runs 1 --single_run --seed $1  --base_lr 0.01 --model_spec maml-reproduce-final --second_order --grad_clip 10 --out_channels 32  --train_iters 240000 --cross_eval 

python -u main.py --problem tiered --k_test 16  --backbone conv4 --model maml --validate --val_after 4000 --T 5 --k 5 --N 5 --T_test 10 --T_val 10 --meta_batch_size 2 --runs 1 --single_run --seed $1  --base_lr 0.01 --model_spec maml-reproduce-final --second_order --grad_clip 10 --out_channels 32  --train_iters 120000 --cross_eval



python -u main.py --problem tiered --k_test 16  --backbone conv4 --model maml --validate --val_after 8000 --T 5 --k 1 --N 5 --T_test 10 --T_val 10 --meta_batch_size 4 --runs 1 --single_run --seed $1  --base_lr 0.01 --model_spec maml-reproduce-final-64c --second_order --grad_clip 10 --out_channels 64  --train_iters 240000 --cross_eval 


python -u main.py --problem tiered --k_test 16  --backbone conv4 --model maml --validate --val_after 4000 --T 5 --k 5 --N 5 --T_test 10 --T_val 10 --meta_batch_size 2 --runs 1 --single_run --seed $1  --base_lr 0.01 --model_spec maml-reproduce-final-64c --second_order --grad_clip 10 --out_channels 64  --train_iters 120000 --cross_eval


**T-Net** (c=32,64): 

python -u main.py --problem tiered --k_test 16  --backbone conv4 --model sap --validate --val_after 8000 --T 5 --k 1 --N 5 --T_test 10 --T_val 10 --meta_batch_size 4 --runs 1 --single_run --seed $1 --base_lr 0.01 --model_spec tnet-reproduce-final-32c --second_order --grad_clip 10 --out_channels 32 --tnet --gamma 0 --reg null  --train_iters 240000 --cross_eval

python -u main.py --problem tiered --k_test 16  --backbone conv4 --model sap --validate --val_after 4000 --T 5 --k 5 --N 5 --T_test 10 --T_val 10 --meta_batch_size 2 --runs 1 --single_run --seed $1 --base_lr 0.01 --model_spec tnet-reproduce-final-32c --second_order --grad_clip 10 --out_channels 32 --tnet --gamma 0 --reg null  --train_iters 120000 --cross_eval


python -u main.py --problem tiered --k_test 16  --backbone conv4 --model sap --validate --val_after 8000 --T 5 --k 1 --N 5 --T_test 10 --T_val 10 --meta_batch_size 4 --runs 1 --single_run --seed $1 --base_lr 0.01 --model_spec tnet-reproduce-final-64c --second_order --grad_clip 10 --out_channels 64 --tnet --gamma 0 --reg null  --train_iters 240000 --cross_eval


python -u main.py --problem tiered --k_test 16  --backbone conv4 --model sap --validate --val_after 4000 --T 5 --k 5 --N 5 --T_test 10 --T_val 10 --meta_batch_size 2 --runs 1 --single_run --seed $1 --base_lr 0.01 --model_spec tnet-reproduce-final-64c --second_order --grad_clip 10 --out_channels 64 --tnet --gamma 0 --reg null  --train_iters 120000 --cross_eval


**MT-Net** (c=32,64):  

python -u main.py --problem tiered --k_test 16  --backbone conv4 --model sap --validate --val_after 8000 --T 5 --k 1 --N 5 --T_test 10 --T_val 10 --meta_batch_size 4 --runs 1 --single_run --seed $1 --base_lr 0.01 --model_spec mtnet-reproduce-final-32c --second_order --grad_clip 10 --out_channels 32 --tnet --gamma 0 --reg null  --train_iters 240000 --use_grad_mask --cross_eval

python -u main.py --problem tiered --k_test 16  --backbone conv4 --model sap --validate --val_after 4000 --T 5 --k 5 --N 5 --T_test 10 --T_val 10 --meta_batch_size 2 --runs 1 --single_run --seed $1 --base_lr 0.01 --model_spec mtnet-reproduce-final-32c --second_order --grad_clip 10 --out_channels 32 --tnet --gamma 0 --reg null  --train_iters 120000 --use_grad_mask --cross_eval



python -u main.py --problem tiered --k_test 16  --backbone conv4 --model sap --validate --val_after 8000 --T 5 --k 1 --N 5 --T_test 10 --T_val 10 --meta_batch_size 4 --runs 1 --single_run --seed $1 --base_lr 0.01 --model_spec mtnet-reproduce-final-64c --second_order --grad_clip 10 --out_channels 64 --tnet --gamma 0 --reg null  --train_iters 240000 --use_grad_mask --cross_eval

python -u main.py --problem tiered --k_test 16  --backbone conv4 --model sap --validate --val_after 4000 --T 5 --k 5 --N 5 --T_test 10 --T_val 10 --meta_batch_size 2 --runs 1 --single_run --seed $1 --base_lr 0.01 --model_spec mtnet-reproduce-final-64c --second_order --grad_clip 10 --out_channels 64 --tnet --gamma 0 --reg null  --train_iters 120000 --use_grad_mask --cross_eval




**WarpGrad**:

python -u main.py --problem tiered --k_test 16  --backbone conv4 --model sap --val_after 2500 --T 5 --k $1 --k_train 5 --N 5 --T_test 5 --T_val 5 --meta_batch_size 1 --runs 1 --single_run --base_lr 0.1 --model_spec warpgrad-reproduce-final-lr0.1-64c --second_order --out_channels 64 --transform_out_channels 64 --tnet --gamma 0 --reg null --warpgrad --use_bias --validate --train_iters 60000  --seed $2 --cross_eval

python -u main.py --problem tiered --k_test 16  --backbone conv4 --model sap --val_after 2500 --T 5 --k $1 --k_train 5 --N 5 --T_test 5 --T_val 5 --meta_batch_size 1 --runs 1 --single_run --base_lr 0.1 --model_spec warpgrad-reproduce-final-lr0.1-32c --second_order --out_channels 32 --transform_out_channels 32 --tnet --gamma 0 --reg null --warpgrad --use_bias --validate --train_iters 60000  --seed $2 --cross_eval


**SAP**:

python -u main.py --problem tiered --N 5 --k $2 --k_test 16 --model sap --model_spec fsap-best-tiered --linear_transform --val_after 2500 --second_order --T 1 --gamma 0 --runs 1 --learn_alfas --reg null --T_test 14 --meta_batch_size 3 --T_val 14 --channel_scale --svd --grad_clip 10 --old --base_lr 0.22697597398238528 --seed $1 --single_run --validate --train_iters 60000 --cross_eval  

python -u main.py --problem tiered --N 5 --k $2 --k_test 16 --model sap --model_spec fsap-best-tiered-c32 --linear_transform --val_after 2500 --second_order --T 1 --gamma 0 --runs 1 --learn_alfas --reg null --T_test 14 --meta_batch_size 3 --T_val 14 --channel_scale --svd --grad_clip 10 --old --base_lr 0.22697597398238528 --seed $1 --single_run --validate --train_iters 60000 --cross_eval --out_channels 32  

python -u main.py --problem tiered --N 5 --k $2 --k_test 16 --model sap --model_spec fsap-best-tiered-c32-FO --linear_transform --val_after 2500 --T 1 --gamma 0 --runs 1 --learn_alfas --reg null --T_test 14 --meta_batch_size 3 --T_val 14 --channel_scale --svd --grad_clip 10 --old --base_lr 0.22697597398238528 --seed $1 --single_run --validate --train_iters 60000 --cross_eval --out_channels 32  


## Questions or feedback?
In case you have any questions or feedback, feel free to reach out to m.huisman@liacs.leidenuniv.nl 
