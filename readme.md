# 分类网络

```mermaid
graph TB
LeNet --deper/bigger--> AlexNet --1x1 conv--> NIN --inception:split-transform-merge--> GoogLeNet --batch normalization --> BN-Inception --updated inception, label smooth --> InceptionV3 --residual connection --> InceptionV4
AlexNet -- 3x3 conv --> VGG -- 1x1 conv bottlenect --> SqueezeNet
VGG -- residual --> ResNet -- + to concat --> DenseNet
VGG --> RepVGG
ResNet -- grouped conv --> ResNext -- shuffle channels among groups --> ShuffleNet
InceptionV3 -- inceeption to depthmise seperate conv --> Xception
ResNet --squeeze-excite --> SENet --> MobileNetV3
MobileNetV1 --> MobileNetV2 --se module --> MobileNetV3
ResNet --Inverted Residual Block--> MobileNetV2
NasNet --> MnasNet
MobileNetV2 --> MnasNet
ResNet --> ResNeSt
MobileNetV3 --> MicroNet
```





# CIFAR10

| Model | Params | FLOPs | Acc(%) |epochs|
|:------:|:------:|:------:|:------:|:------:|
|VGG16|-|-|93.61|150|
|ResNet18| 11.18M | 37.12M | 87.95 |150|
|MobileNetv1| 3.22M | 11.80M | 87.98 |150|



```bash
python cifar_train.py --arch resnet --cfg resnet18_cifar --data_path F:\Datasets\CIFAR10 --job_dir ./resnet --lr 0.01 --lr_decay_step 50 100 --weight_decay 0.005 --num_epochs 150 --gpus 0
```