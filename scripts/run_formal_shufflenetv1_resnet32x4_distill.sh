# sample scripts for running the distillation code
# use resnet32x4 and resnet8x4 as an example

# 定义一个变量名为
declare -i trial=1
declare -i exc_epoch=8

# kd
python train_formal_shufflenetv1_resnet32x4_exc.py --path-t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --mindex=1 --distill kd --model_s ShuffleV1 --exc_epoch=$exc_epoch -r 1 -a 1 -b 0 --trial $trial
# SemCKD
python train_formal_shufflenetv1_resnet32x4_exc.py --path-t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --mindex=2 --distill semckd --model_s ShuffleV1 --exc_epoch=$exc_epoch -r 1 -a 1 -b 400 --trial $trial
# RKD
python train_formal_shufflenetv1_resnet32x4_exc.py --path-t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --mindex=3 --distill rkd --model_s ShuffleV1 --exc_epoch=$exc_epoch -r 1 -a 1 -b 1 --trial $trial
# CRD
python train_formal_shufflenetv1_resnet32x4_exc.py --path-t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --mindex=4 --distill crd --model_s ShuffleV1 --exc_epoch=$exc_epoch -r 1 -a 1 -b 0.8 --trial $trial
