
import torchvision
import torch.nn as nn
import torch

import timm

## simple wrapper model to normalize an input image
class WrapperModel(nn.Module):
    def __init__(self, model, mean, std,resize=False):
        super(WrapperModel, self).__init__()
        self.mean = torch.Tensor(mean)
        self.model=model
        self.resize=resize
        self.std = torch.Tensor(std)
    def forward(self, x):
        return self.model((x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None])



def load_model(model_name):
    if model_name == "ResNet101":
        model = torchvision.models.resnet101(pretrained=True)
    elif model_name == 'ResNet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif model_name == 'ResNet34':
        model = torchvision.models.resnet34(pretrained=True)
    elif model_name == 'ResNet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif model_name == "ResNet152":
        model = torchvision.models.resnet152(pretrained=True)
    elif model_name == "vgg16":
        model = torchvision.models.vgg16_bn(pretrained=True)
    elif model_name == "vgg19":
        model = torchvision.models.vgg19_bn(pretrained=True)
    elif model_name == "wide_resnet101_2":
        model = torchvision.models.wide_resnet101_2(pretrained=True)
    elif model_name == "inception_v3":
        model = torchvision.models.inception_v3(pretrained=True,transform_input=True)
    elif model_name == "resnext50_32x4d":
        model = torchvision.models.resnext50_32x4d(pretrained=True) 
    elif model_name == "alexnet":
        model = torchvision.models.alexnet(pretrained=True)
    elif model_name == "mobilenet_v3_large":
        model = torchvision.models.mobilenet.mobilenet_v3_large(pretrained=True)
    elif model_name == 'DenseNet121':
        model = torchvision.models.densenet121(pretrained=True)
    elif model_name == "DenseNet161":
        model = torchvision.models.densenet161(pretrained=True)
    elif model_name == 'mobilenet_v2':
        model = torchvision.models.mobilenet_v2(pretrained=True)
    elif model_name == "shufflenet_v2_x1_0":
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
    elif model_name == 'GoogLeNet':
        model = torchvision.models.googlenet(pretrained=True)
    # timm models
    elif model_name == "adv_inception_v3":
        model = timm.create_model("adv_inception_v3", pretrained=True)
    elif model_name == "inception_resnet_v2":
        model = timm.create_model("inception_resnet_v2", pretrained=True)
    elif model_name == "ens_adv_inception_resnet_v2":
        model = timm.create_model("ens_adv_inception_resnet_v2", pretrained=True)
    elif model_name == "inception_v3_timm":
        model = timm.create_model("inception_v3", pretrained=True)
    elif model_name == "inception_v4_timm":
        model = timm.create_model("inception_v4", pretrained=True)
    elif model_name == "xception":
        model = timm.create_model("xception", pretrained=True)
    else:
        raise ValueError(f"Not supported model name. {model_name}")
    return model