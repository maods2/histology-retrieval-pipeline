from mmengine.model import BaseModel

test_param = dict(
    model_param = dict(
        output_exp = dict(sclerosis = "./embedding_encoder/src/models/sclerosis_epoch_50.pth",
                          glomerulus = "./embedding_encoder/src/models/glomerulus_epoch_50.pth"),
        filename_checkpoint = "epoch_50.pth",
        encoder_name = "efficientnet-b0",
        encoder_weights = "imagenet",
        classes = 1,
        activation = None,
        network_name = "unet",
    ),
    model = "unet",
    input_resolution = 320,
    resolution = 1024,
    pad_size = 0,
    clf_threshold = 0.5,
    small_mask_threshold = 0,
    mask_threshold = 0.5,
    tta = 3,
    test_batch_size = 12,
    num_workers = 4,
) 


class ConfigClass:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class _Model(BaseModel):
    def __init__(self,model):
        super().__init__()
        self.model = model

    def forward(self, imgs, labels, mode):
        x = self.model(imgs)
        if mode == 'loss':
            return 0 #{'loss': criterion(x, labels)}

        elif mode == 'predict':
            return x, labels