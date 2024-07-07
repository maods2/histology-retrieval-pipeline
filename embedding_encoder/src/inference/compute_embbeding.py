from utils.dataset import ImageDataLoader
from models.model_selector import get_model
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import torch
import pickle 
# from options import BaseOptions, Config, load_parameters
from utils.utils import slice_image_paths, generate_checkpoint_file
from tqdm import tqdm

def load_model_weights(model, model_path,device):
    checkpoint = torch.load(model_path,map_location=torch.device(device))
    model.load_state_dict(checkpoint['state_dict'])
    return model

def compute_embeddings(config, mode, model=None):
    # Load data
    if mode == "train":
        data = ImageDataLoader(config.data_path, config)
    elif mode == "test":
        data = ImageDataLoader(config.val_data_path, config)
    else:
        raise Exception("Embedding computation mode not set")

    dataloader = DataLoader(data.dataset, batch_size=config.batch_size, shuffle=False)

    # select embedding model training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    if model == None:
        model = get_model(config.model)
        if config.load_checkpoint:
            checkpoint_file = generate_checkpoint_file(config.model_name, search_dir=config.checkpoint_dir)
            model = load_model_weights(model,checkpoint_file , device)

    model.eval()
    model.to(device)

    # compute embeddings and save
    target = []
    paths = []
    labels = []
    feature_embeddings = np.empty((0, config.embedding_dim))

    for i, (x, y, path, label) in enumerate(tqdm(dataloader)):
        x = x.to(device=device)
        with torch.no_grad():
            batch_features = model(x)

        batch_features = batch_features.view(batch_features.size(0), -1).cpu().numpy()
        feature_embeddings = np.vstack((feature_embeddings, batch_features))
        target.extend(list(y.cpu().detach().numpy()))
        paths.extend(slice_image_paths(path))
        labels.extend(label)


    data_dict = {
        "model": config.model,
        "embedding":feature_embeddings,
        "target":target,
        "paths": paths,
        "classes":labels
    }
    path = f'{config.save_embedding_path}/{config.prefix}'
    Path(path).mkdir(parents=True, exist_ok=True)

    with open(f'{path}/{config.model_name}_{mode}.pickle', 'wb') as pickle_file:
        pickle.dump(data_dict, pickle_file)
