import torch.nn as nn
import torchvision.models as models
import timm


# import torch
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(DEVICE)
#
# dummy_img = torch.zeros((1, 3, 800, 800)).float()   # test image array
#
# req_features = []
# output = dummy_img.clone().to(DEVICE)

# model1 = models.efficientnet_b0(pretrained=True).to(DEVICE)
# model1 = models.resnet18(pretrained=True).to(DEVICE)
# model1 = models.vgg16(pretrained=True).to(DEVICE)
# print(timm.list_models('eff*'))
# model2 = timm.create_model('efficientnet_b0').to(DEVICE)
# print(model1)
# print(model1.classifier[1].in_features)
# print(list(model1.children())[:-2])
# in_channel = list(model1.children())[-2].out_features
# model = nn.Sequential(*list(model1.children())[:-2])
# print(model)
# for name, layer in enumerate(model1.children()):
#     print('{} is {}'.format(name, layer))


import torch
import torch.nn as nn
import torchvision.models as models


class my_model(nn.Module):
    def __init__(self,
                 num_class,
                 init_weights=True):
        super(my_model, self).__init__()

        # self.pretrained = models.vgg16(pretrained=True)
        self.split_model = self.split_pretrained()
        # print(self.split_model)
        self.avg_pool = nn.AvgPool2d(kernel_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(63, num_class)
        )

        if init_weights:
            self.initialize_weight()

    def split_pretrained(self):
        return nn.Sequential(*list(models.vgg16(pretrained=True).children())[:-2])

    def initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.split_model(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


model = my_model(23)
print(model)
