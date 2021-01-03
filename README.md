# Pytorch.SSD
A strong implement of SSD ~

## Pytorch.SSD v1 Release （2020.1.3）
1、The nms is update to torchvision cuda nms, which is much fasater when eval and test
2、Model init is update to xavier_uniform_, which makes the model converge more faster
3、Warmup lr is added
4、Good performance:
|method|train data|test data|mAP|aero|bike|bird|boat|bottle|bus|car|cat|chair|cow|table|dog|horse|mbike|person|plant|sheep|sofa|train|tv|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|SSD (Liu et al.)|VOC 2007 trainval|VOC 2007 test|0.680|73.4|77.5|64.1|59.0|38.9|75.2|80.8|78.5|46.0|67.8|69.2|76.6|82.1|77.0|72.5|41.2|64.2|69.1|78.0|68.5|
|SSD (ours)|VOC 2007 trainval|VOC 2007 test|0.7240|0.7761|0.8111|0.6898|0.6253|0.4399|0.8108|0.8368|0.8259|0.5456|0.7754|0.7227|0.7991|0.8242|0.7956|0.7679|0.4565|0.6967|0.7368|0.8300|0.7228|

## Pytorch.SSD v0 Release （2020.12.31）
1、The model can be trained with eval
2、The cfg paras is defined in yaml file, which is then merged with args.opts
