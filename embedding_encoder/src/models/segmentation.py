import torch
from torch import nn
import segmentation_models_pytorch as smp
from embedding_encoder.src.models.config import ConfigClass, _Model, test_param
from mmengine.runner.checkpoint import load_checkpoint
from pathlib  import Path


class SclerosisSegmentationModel(nn.Module):
    def __init__(self, config: ConfigClass):
        super().__init__()
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.glom_model = self._create_model().to(self.device )
        self.scler_model = self._create_model().to(self.device )
        
        self.glom_model = _Model(self.glom_model)
        self.scler_model = _Model(self.scler_model)
        
        self._load_checkpoints()

    def _create_model(self):
        return smp.Unet(
            encoder_name=self.cfg.model_param['encoder_name'],
            encoder_weights=self.cfg.model_param['encoder_weights'],
            classes=self.cfg.model_param['classes'],
            activation=self.cfg.model_param['activation'],
        )

    def _load_checkpoints(self):
        checkpoint_glom = Path(self.cfg.model_param['output_exp']['glomerulus'])
        checkpoint_scle = Path(self.cfg.model_param['output_exp']['sclerosis'])
        load_checkpoint(self.glom_model, checkpoint_glom.as_posix())
        load_checkpoint(self.scler_model, checkpoint_scle.as_posix())
        self.glom_model = self.glom_model.model
        self.scler_model = self.scler_model.model

    def forward(self, img_patch):
        glom_mask = torch.sigmoid(self.glom_model(img_patch))
        scler_mask = torch.sigmoid(self.scler_model(img_patch))
        inter = torch.logical_and(glom_mask > 0.5, scler_mask > 0.5)
        p = torch.sum(inter, dim=(2, 3), keepdim=True).float() / (torch.sum(glom_mask > 0.5, dim=(2, 3), keepdim=True).float() + 0.00001)
        p = p.view(p.size(0), -1)
        return p


if "__main__" == __name__:

    height = 256
    width = 256
    cfg = ConfigClass(**test_param)
    model = SclerosisSegmentationModel(cfg)
    img_batch = torch.randn(15, 3, height, width).to(model.device)
    p= model(img_batch)
    print(p)
    print(p.shape)




