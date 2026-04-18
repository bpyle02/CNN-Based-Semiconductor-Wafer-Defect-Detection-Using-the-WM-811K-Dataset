import torch
import torch.nn as nn
import torchvision.models as models

class WaferHybridModel(nn.Module):
    def __init__(self, model_name="resnet18", num_classes=9, num_geom_features=6, pretrained=True):
        super(WaferHybridModel, self).__init__()
        
        # 1. Image Backbone
        if model_name == "resnet18":
            self.backbone = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity() # Remove the last FC layer
        elif model_name == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
            num_ftrs = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Identity() # Remove the last FC layer
        else:
            raise ValueError(f"Model {model_name} not supported")

        # 2. Geometric Feature Branch
        self.geom_fc = nn.Sequential(
            nn.Linear(num_geom_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 3. Final Combined Classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        img, geom = x
        img_features = self.backbone(img)
        geom_features = self.geom_fc(geom)
        
        # Concatenate
        combined = torch.cat((img_features, geom_features), dim=1)
        return self.classifier(combined)

def get_model(model_name="resnet18", num_classes=9, num_geom_features=6, pretrained=True):
    return WaferHybridModel(model_name=model_name, num_classes=num_classes, 
                            num_geom_features=num_geom_features, pretrained=pretrained)
