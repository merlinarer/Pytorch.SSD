# Pytorch.SSD
A strong implement of SSD ~

## Pytorch.SSD v1 Release （2020.1.3）
1、The nms is update to torchvision cuda nms, which is much fasater when eval and test

2、Model init is update to xavier_uniform_, which makes the model converge more faster

3、Warmup lr is added

4、Good performance

|method|train data|test data|mAP|aero|bike|bird|boat|bottle|bus|car|cat|chair|cow|table|dog|horse|mbike|person|plant|sheep|sofa|train|tv|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|SSD (Liu et al.)|VOC 2007 trainval|VOC 2007 test|0.680|0.734|0.775|0.641|0.590|0.389|0.752|0.808|0.785|0.460|0.678|0.692|0.766|0.821|0.770|0.725|0.412|0.642|0.691|0.780|0.685|
|SSD (mmdet)|VOC 2007 trainval|VOC 2007 test|0.709|0.765|0.798|0.669|0.628|0.413|0.800|0.820|0.801|0.513|0.735|0.669|0.803|0.829|0.788|0.742|0.438|0.674|0.715|0.833|0.710|
|SSD (ours)|VOC 2007 trainval|VOC 2007 test|0.7240|0.7761|0.8111|0.6898|0.6253|0.4399|0.8108|0.8368|0.8259|0.5456|0.7754|0.7227|0.7991|0.8242|0.7956|0.7679|0.4565|0.6967|0.7368|0.8300|0.7228|

## Pytorch.SSD v0 Release （2020.12.31）
1、The model can be trained with eval

2、The cfg paras is defined in yaml file, which is then merged with args.opts

## References
> Wei Liu, et al. "SSD: Single Shot MultiBox Detector." ECCV2016.

> ssd.pytorch (https://github.com/amdegroot/ssd.pytorch)

> mmdetection (https://github.com/open-mmlab/mmdetection)
