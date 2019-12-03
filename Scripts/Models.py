import torchvision.models as models
import cv2
import torch
import numpy as np
from Settings import Settings
import torch.nn as nn

# vgg16 = models.vgg16_bn(pretrained=True)
vgg16 = models.vgg16_bn()
state_dict = torch.utils.model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth')
vgg16.load_state_dict(state_dict)
vgg16_conv = list(vgg16.children())[0]


def get_image_feature(im, model=vgg16_conv):
    model = model.eval()
    with torch.no_grad():
        image = cv2.resize(im, (224, 224)).astype(np.float32)
        image = torch.from_numpy(image).view((1, 3, 224, 224))

    if Settings.cuda:
        model = model.cuda()
        image = image.cuda()

    feature_out = model(image)
    return feature_out.view(1, -1).detach() # no need to record 


class QModel(nn.Module):
    def __init__(self):
        super(QModel, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(25088 + 24, 1024),
                nn.ReLU(),
                nn.Dropout(0.6),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(0.6),
                nn.Linear(1024, 6),  # six movement
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    from FileUtils import FileUtils
    data = FileUtils(3)
    feature = get_image_feature(data.images[0])
    print()

