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

nohup python -u main.py --problem sine --k 5 --k_test 50 --model maml --model_spec fmaml --cpu --val_after 2500 --second_order --T 1 --runs 5 --validate --meta_batch_size 4 > saplogs/sine/sine-maml-k5.log &

nohup python -u main.py --problem sine --k 10 --k_test 50 --model maml --model_spec fmaml --cpu --val_after 2500 --second_order --T 1 --runs 5 --validate --meta_batch_size 4 > saplogs/sine/sine-maml-k10.log &


**T-Net**

nohup python -u main.py --problem sine --k 5 --k_test 50 --model sap --model_spec ftnet --cpu --val_after 2500 --second_order --T 1 --gamma 0 --runs 5 --reg null --validate --meta_batch_size 4 --tnet > saplogs/sine/sine-tnet-k5.log &
 
nohup python -u main.py --problem sine --k 10 --k_test 50 --model sap --model_spec ftnet --cpu --val_after 2500 --second_order --T 1 --gamma 0 --runs 5 --reg null --validate --meta_batch_size 4 --tnet > saplogs/sine/sine-tnet-k10.log &


**MT-Net**

nohup python -u main.py --problem sine --k 5 --k_test 50 --model sap --model_spec fmtnet --cpu --val_after 2500 --second_order --T 1 --gamma 0 --runs 5 --reg null --validate --meta_batch_size 4 --tnet --use_grad_mask > saplogs/sine/sine-mtnet-k5.log &
 
nohup python -u main.py --problem sine --k 10 --k_test 50 --model sap --model_spec fmtnet --cpu --val_after 2500 --second_order --T 1 --gamma 0 --runs 5 --reg null --validate --meta_batch_size 4 --tnet --use_grad_mask > saplogs/sine/sine-mtnet-k10.log &


**SAP**

nohup python -u main.py --problem sine --k 5 --k_test 50 --model sap --model_spec fsap-svd --cpu --val_after 2500 --second_order --T 1 --gamma 0 --runs 5 --learn_alfas --reg null --svd --validate --meta_batch_size 4 > saplogs/sine/sine-sap-svd-k5.log &
 
nohup python -u main.py --problem sine --k 10 --k_test 50 --model sap --model_spec fsap-svd --cpu --val_after 2500 --second_order --T 1 --gamma 0 --runs 5 --learn_alfas --reg null --svd --validate --meta_batch_size 4 > saplogs/sine/sine-sap-svd-k10.log &


nohup python -u main.py --problem sine --k 5 --k_test 50 --model sap --model_spec fmsap-svd --cpu --val_after 2500 --second_order --T 1 --gamma 0 --runs 5 --learn_alfas --reg null --svd --validate --meta_batch_size 4 --use_grad_mask > saplogs/sine/sine-msap-svd-k5.log &
 
nohup python -u main.py --problem sine --k 10 --k_test 50 --model sap --model_spec fmsap-svd --cpu --val_after 2500 --second_order --T 1 --gamma 0 --runs 5 --learn_alfas --reg null --svd --validate --meta_batch_size 4 --use_grad_mask > saplogs/sine/sine-msap-svd-k10.log &


### Few-shot image classification



## Questions or feedback?
In case you have any questions or feedback, feel free to reach out to m.huisman@liacs.leidenuniv.nl 
