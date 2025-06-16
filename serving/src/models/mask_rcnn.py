
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch.nn as nn
import torch

def create_mask_rcnn_model(num_classes, pretrained=False, trainable_backbone_layers=3, **kwargs):

    model = maskrcnn_resnet50_fpn(
        pretrained=pretrained,
        pretrained_backbone=pretrained,
        trainable_backbone_layers=trainable_backbone_layers,
        **kwargs
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_masks = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_masks,
        hidden_layer,
        num_classes
    )
    return model
