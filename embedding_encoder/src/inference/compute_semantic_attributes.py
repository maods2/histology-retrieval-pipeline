from datetime import datetime
import sys

# from utils.dataset import ImageDataLoader
from torch.utils.data import DataLoader
# from utils.utils import slice_image_paths
import matplotlib.pyplot as plt
import pickle
import torch
import torchvision
from efficientnet_pytorch import EfficientNet
import numpy as np
import re
from PIL import Image
from torch import nn
from torchvision import transforms
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from typing import Callable, Optional, Tuple, Any
import os
from pathlib import Path
import sys
# print(sys.path)
sys.path.append('/home/ivisionlab/histology-retrieval-pipeline/')
from embedding_encoder.src.models.config import ConfigClass, _Model, test_param
from embedding_encoder.src.models.segmentation import SclerosisSegmentationModel
from mmengine.runner.checkpoint import load_checkpoint

def slice_image_paths(paths):
    return [i.split('/')[-1].replace('\\','/') for i in paths]

class ImageDataLoader:
    def __init__(self, data_dir, config):
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.mean, std=config.std)
        ])
        self.dataset = CustomImageFolder(
            self.data_dir, transform=self.transform, target_transform=self._get_class_name)

        self.dataloader = DataLoader(self.dataset, shuffle=False)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return iter(self.dataloader)

    def _get_class_name(self, index):
        return index


class CustomImageFolder(ImageFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):

        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
        )
        # self.paths = [s[0] for s in self.samples]
        # self.labels = [self.classes[s[1]]
        #                for s in self.samples]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path, self.classes[target]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def load_model_weights(model, model_path,device):
    checkpoint = torch.load(model_path,map_location=torch.device(device))
    model.load_state_dict(checkpoint['state_dict'])
    return model

def get_all_image_files(pathlib_root_folder):
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    img_regex = re.compile('|'.join(img_extensions), re.IGNORECASE)
    image_files = [f for f in pathlib_root_folder.glob('**/*') if f.is_file() and img_regex.search(f.suffix)]
    return image_files


def get_model(model_name, settings, freeze=False):
    if model_name == "efficientnet":
        return Net(net_version="b0", num_classes=2, freeze=freeze).to(settings.device)
    elif model_name == "dino":
       return Dino(num_classes=2, settings=settings, freeze=freeze).to(settings.device)
    else:
        raise Exception("None model defined")



class Net(nn.Module):
    def __init__(self, net_version, num_classes, freeze: bool = False):
        super(Net, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-'+net_version)
        self.backbone._fc = nn.Sequential(
            nn.Linear(1280, num_classes),
        )
        if freeze:
            # freeze backbone layers
            for name, param in self.backbone.named_parameters():
                if not name.startswith("_fc"):
                    param.requires_grad = False

    def forward(self, x):
        return self.backbone(x)

class Dino(nn.Module):
    def __init__(self, num_classes, settings, freeze: bool = False):
        super(Dino, self).__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc')
        self.linear = nn.Sequential(
            nn.Linear(settings.model.fcLayer, num_classes),
        )
        if freeze:
            # freeze backbone layers
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.linear(self.backbone(x))

def compute_semantic_attributes(config, mode):

    models = []
    for filename in config.semantic_models_path:
        path = config.semantic_basepath
        model_files = [os.path.join(path, f) for f in os.listdir(path)]
        model_path = [f for f in model_files if filename in f][0]
        models.append(
            load_model_weights(get_model(config.model, config), model_path, device)
        )
    
    if config.sclerosis_segmentation:
        cfg = ConfigClass(**test_param)
        segment = SclerosisSegmentationModel(cfg)

    data = ImageDataLoader(config.data_path, config)
    dataloader = DataLoader(data.dataset, batch_size=config.batch_size, shuffle=False)

    target = []
    paths = []
    labels = []
    num_att = config.num_att
    feature_embeddings = np.empty((0, num_att))






    for i, (x, y, path, label) in enumerate(dataloader):
        x = x.to(device=device)
        with torch.no_grad():
            prediction_columns = []
            for model in models:
                model.eval()
                output = model(x)
                scores = torch.softmax(output,dim=-1)
                prediction_columns.append(scores[:, 0].view(-1, 1))
            
            if config.sclerosis_segmentation:
                p = segment(x)
                prediction_columns.append(p)



        prediction_matrix = torch.cat(prediction_columns, dim=1)

          
        emb_batch = prediction_matrix.cpu().detach().numpy()
        feature_embeddings = np.vstack((feature_embeddings, emb_batch))
        target.extend(list(y.cpu().detach().numpy()))
        paths.extend(slice_image_paths(path))
        labels.extend(label)

        print(f"{i} of {len(dataloader)} batchs")

    data_dict = {
        "model": 'semantic_att',
        "embedding":feature_embeddings,
        "target":target,
        "paths": paths,
        "classes":labels
    }



    path = f'{config.save_embedding_path}/'
    Path(path).mkdir(parents=True, exist_ok=True)

    with open(f'{path}/{config.model}_{mode}.pickle', 'wb') as pickle_file:
        pickle.dump(data_dict, pickle_file)



if __name__ == "__main__":
    from dynaconf import Dynaconf

    config_path = "embedding_encoder/config/inference_semantic_attributes_test.toml"
    config = Dynaconf(
        envvar_prefix="DYNACONF",
        settings_file=config_path,
        root_path='../..',
    )
    compute_semantic_attributes(config,'train')