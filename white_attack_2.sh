# --------- CIFAR10-vgg/resnet -----------------
# --------- OMP-a
# arch=OMPa
# ----
# model=vgg16
# model_path='./save/CIFAR10-OMPa-vgg16-lamb-0.1-path-10-adv.pth'
# ----
# model=resnet20
# model_path='./save/CIFAR10-OMPa-resnet20-lamb-0.1-path-10-adv.pth'
# --------- OMP-b
arch=OMPb
# ----
# model=vgg16
# model_path='./save/CIFAR10-OMPb-vgg16-lamb-0.1-path-10-adv.pth'
# ----
model=resnet20
model_path='./save/CIFAR10-OMPb-resnet20-lamb-0.1-path-10-adv.pth'
# -------- hyper-parameters ---------------------
num_paths=10
selected_path=5
# -------- CIFAR10 ------------------------------
dataset=CIFAR10
data_dir='/media/Disk1/KunFang/data/CIFAR10/'
# -----------------------------------------------
gpu_id=0

python white_attack_2.py \
    --arch ${arch} \
    --model ${model} \
    --model_path ${model_path} \
    --dataset ${dataset} \
    --data_dir ${data_dir}   \
    --gpu_id ${gpu_id} \
    --num_paths ${num_paths} \
    --selected_path ${selected_path}
