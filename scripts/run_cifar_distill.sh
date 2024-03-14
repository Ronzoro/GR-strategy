# sample scripts for running the distillation code
# use resnet32x4 and resnet8x4 as an example

# kd
python train_student.py --path-t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 1 -a 1 -b 0 --trial 1
# RKD
python train_student.py --path-t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill rkd --model_s resnet8x4 -r 1 -a 1 -b 1 --trial 1
# CRD
python train_student.py --path-t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet8x4 -r 1 -a 1 -b 0.8 --trial 1
# SemCKD
python train_student.py --path-t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill semckd --model_s resnet8x4 -r 1 -a 1 -b 400 --trial 1
