# --------- CIFAR10-vgg/resnet -----------------
# --------- OMP-a
# arch=OMPa
# ----
# model=vgg16
# model_path='./save/CIFAR10-OMPa-vgg16-lamb-0.1-path-10.pth'
# model_path='./save/CIFAR10-OMPa-vgg16-lamb-0.1-path-10-adv.pth'
# ----
# model=resnet20
# model_path='./save/CIFAR10-OMPa-resnet20-lamb-0.1-path-10.pth'
# model_path='./save/CIFAR10-OMPa-resnet20-lamb-0.1-path-10-adv.pth'
# --------- OMP-b
# arch=OMPb
# ----
# model=vgg16
# model_path='./save/CIFAR10-OMPb-vgg16-lamb-0.1-path-10.pth'
# model_path='./save/CIFAR10-OMPb-vgg16-lamb-0.1-path-10-adv.pth'
# ----
# model=resnet20
# model_path='./save/CIFAR10-OMPb-resnet20-lamb-0.1-path-10.pth'
# model_path='./save/CIFAR10-OMPb-resnet20-lamb-0.1-path-10-adv.pth'
# --------- OMP-c
arch=OMPc
# ----
model=vgg16
# model_path='./save/CIFAR10-OMPc-vgg16-lamb-0.1-path-10.pth'
model_path='./save/CIFAR10-OMPc-vgg16-lamb-0.1-path-10-adv.pth'
# ----
# model=resnet20
# model_path='./save/CIFAR10-OMPc-resnet20-lamb-0.1-path-10.pth'
# model_path='./save/CIFAR10-OMPc-resnet20-lamb-0.1-path-10-adv.pth'
# -------- hyper-parameters ---------------------
num_paths=10
# -------- CIFAR10 ------------------------------
dataset=CIFAR10
data_dir='/media/Disk1/KunFang/data/CIFAR10/'
# -----------------------------------------------
gpu_id=0

# -----------------------------------------------
source_model=vgg16
source_model_path='./save/CIFAR10-vgg16.pth'
# source_model=resnet20
# source_model_path='./save/CIFAR10-resnet20.pth'
# -----------------------------------------------

python black_attack.py \
    --arch ${arch} \
    --model ${model} \
    --model_path ${model_path} \
    --dataset ${dataset} \
    --data_dir ${data_dir}   \
    --gpu_id ${gpu_id} \
    --num_paths ${num_paths} \
    --source_model ${source_model} \
    --source_model_path ${source_model_path}
