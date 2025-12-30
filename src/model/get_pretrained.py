from torchgeo.models import ResNet50_Weights, resnet50

def torchgeo_res50():
    """_summary_
    Load pretrained model weights.
    """
    rgb_weights = ResNet50_Weights.SENTINEL2_RGB_MOCO
    model = resnet50(rgb_weights)
    return model