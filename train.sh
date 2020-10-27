# -------- architecture -------------------------
arch=OMP_a
# arch=OMP_b
# arch=OMP_c
# -------- model:vgg16/resnet18 -----------------
# model=vgg11
# model=vgg13
model=vgg16
# model=vgg19
# model=resnet20
# model=resnet32
# -------- hyper-parameters ---------------------
lamb=0.01
num_paths=10
# -------- CIFAR10 ------------------------------
dataset=CIFAR10
data_dir='/media/Disk1/KunFang/data/CIFAR10/'
# -------- model directory ----------------------
model_dir='./save/'
# -----------------------------------------------
gpu_id=2
# -----------------------------------------------
adv_train=False
# adv_train=True
# -----------------------------------------------

python train.py \
    --arch ${arch} \
    --model ${model} \
    --lamb ${lamb} \
    --num_paths ${num_paths} \
    --dataset ${dataset} \
    --data_dir ${data_dir} \
    --model_dir ${model_dir} \
    --gpu_id ${gpu_id} \
    --adv_train ${adv_train}
