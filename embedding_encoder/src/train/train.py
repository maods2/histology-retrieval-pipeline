from models.model_selector import get_model
from pipelines.pipeline_selector import get_pipeline
from utils.utils import save_checkpoint
from inference.compute_embbeding import compute_embeddings
import numpy as np




def train_and_save_embeddings(config):

    model = get_model(config.model)
    train_model = get_pipeline(config.pipeline)

    optimizer, loss = train_model(
        model=model,
        config=config
    )
    save_checkpoint(model, optimizer, loss, config)
    # compute embeddings and save

    compute_embeddings(config,"train", model)
    compute_embeddings(config,"test", model)

   
