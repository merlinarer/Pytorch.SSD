# Pytorch.SSD
This repo is a strong Pytorch implementation of SSD ~

## [Pytorch.SSD v1 Release](https://github.com/merlinarer/Pytorch.SSD/releases/tag/v1)

- Update the nms to torchvision cuda one, which is much fasater
- Update model init to xavier_uniform_, making the model converge more faster
- Warmup lr is added
- Good performance

|method|train data|test data|mAP|aero|bike|bird|boat|bottle|bus|car|cat|chair|cow|table|dog|horse|mbike|person|plant|sheep|sofa|train|tv|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|SSD (Liu et al.)|VOC 2007 trainval|VOC 2007 test|0.680|0.734|0.775|0.641|0.590|0.389|0.752|0.808|0.785|0.460|0.678|0.692|0.766|0.821|0.770|0.725|0.412|0.642|0.691|0.780|0.685|
|SSD (mmdet)|VOC 2007 trainval|VOC 2007 test|0.709|0.765|0.798|0.669|0.628|0.413|0.800|0.820|0.801|0.513|0.735|0.669|0.803|0.829|0.788|0.742|0.438|0.674|0.715|0.833|0.710|
|SSD (ours)|VOC 2007 trainval|VOC 2007 test|0.7240|0.7761|0.8111|0.6898|0.6253|0.4399|0.8108|0.8368|0.8259|0.5456|0.7754|0.7227|0.7991|0.8242|0.7956|0.7679|0.4565|0.6967|0.7368|0.8300|0.7228|

## [Pytorch.SSD v0 Release](https://github.com/merlinarer/Pytorch.SSD/releases/tag/v0)
- The model can be trained smoothly
- The cfg paras is defined in yaml file, which is then merged with args.opts

## References
> Wei Liu, et al. "SSD: Single Shot MultiBox Detector." ECCV2016.

> [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)
> [mmdetection](https://github.com/open-mmlab/mmdetection)
