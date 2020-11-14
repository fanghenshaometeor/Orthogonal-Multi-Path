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
# model=vgg11
# model_path='./save/CIFAR10-OMPc-vgg11-lamb-0.1-path-10.pth'
# model_path='./save/CIFAR10-OMPc-vgg11-lamb-0.1-path-10-adv.pth'
# ----
# model=vgg13
# model_path='./save/CIFAR10-OMPc-vgg13-lamb-0.1-path-10.pth'
# model_path='./save/CIFAR10-OMPc-vgg13-lamb-0.1-path-10-adv.pth'
# ----
# model=vgg16
# model_path='./save/CIFAR10-OMPc-vgg16-lamb-0.1-path-10.pth'
# model_path='./save/CIFAR10-OMPc-vgg16-lamb-0.1-path-10-adv.pth'
# model_path='./save/CIFAR100-OMPc-vgg16-lamb-0.1-path-10.pth'
# model_path='./save/CIFAR100-OMPc-vgg16-lamb-0.1-path-10-adv.pth'
# ----
# model=vgg19
# model_path='./save/CIFAR10-OMPc-vgg19-lamb-0.1-path-10.pth'
# model_path='./save/CIFAR10-OMPc-vgg19-lamb-0.1-path-10-adv.pth'
# ----
# model=resnet20
# model_path='./save/CIFAR10-OMPc-resnet20-lamb-0.1-path-10.pth'
# model_path='./save/CIFAR10-OMPc-resnet20-lamb-0.1-path-10-adv.pth'
# model_path='./save/CIFAR100-OMPc-resnet20-lamb-0.1-path-10.pth'
# model_path='./save/CIFAR100-OMPc-resnet20-lamb-0.1-path-10-adv.pth'
# ----
# model=resnet32
# model_path='./save/CIFAR10-OMPc-resnet32-lamb-0.1-path-10.pth'
# model_path='./save/CIFAR10-OMPc-resnet32-lamb-0.1-path-10-adv.pth'
# ----
model=modela
# model_path='./save/STL10-OMPc-modela-lamb-0.1-path-10.pth'
model_path='./save/STL10-OMPc-modela-lamb-0.1-path-10-adv.pth'
# -------- hyper-parameters ---------------------
num_paths=10
# -------- CIFAR10 ------------------------------
# dataset=CIFAR10
# data_dir='/media/Disk1/KunFang/data/CIFAR10/'
# -------- CIFAR100 -----------------------------
# dataset=CIFAR100
# data_dir='/media/Disk1/KunFang/data/CIFAR100/'
# -------- STL10 --------------------------------
dataset=STL10
data_dir='/media/Disk1/KunFang/data/STL10/'
# -----------------------------------------------
gpu_id=0

python test.py \
    --arch ${arch} \
    --model ${model} \
    --model_path ${model_path} \
    --dataset ${dataset} \
    --data_dir ${data_dir}   \
    --gpu_id ${gpu_id} \
    --num_paths ${num_paths}