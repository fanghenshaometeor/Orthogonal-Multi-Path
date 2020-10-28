# -------- architecture -------------------------
arch=OMPc
# -------- model:vgg16/resnet18 -----------------
model=resnet20
# -------- hyper-parameters ---------------------
lamb=0
num_paths=10
# -------- CIFAR10 ------------------------------
dataset=CIFAR10
data_dir='/media/Disk1/KunFang/data/CIFAR10/'
# -------- model directory ----------------------
model_dir='./save/'
# -----------------------------------------------
gpu_id=2
# -----------------------------------------------
adv_train=True
# -----------------------------------------------

python train_ablation.py \
    --arch ${arch} \
    --model ${model} \
    --lamb ${lamb} \
    --num_paths ${num_paths} \
    --dataset ${dataset} \
    --data_dir ${data_dir} \
    --model_dir ${model_dir} \
    --gpu_id ${gpu_id} \
    --adv_train ${adv_train}
