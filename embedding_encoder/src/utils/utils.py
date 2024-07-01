from pathlib import Path
import torch
import time
import matplotlib.pyplot as plt
import yaml

import os
import re

def find_last_checkpoint_dir(class_name: str, search_dir: str) -> str:
    all_checkpoint_dirs = os.listdir(search_dir)
    class_checkpoints = list(filter(lambda path: class_name in path, all_checkpoint_dirs))
    class_checkpoints = list(filter(lambda path: os.path.isdir(os.path.join(search_dir, path)),
                                    class_checkpoints))  # filter for directories only
    try:
        print("Using checkpoint folder: ", sorted(class_checkpoints)[-1])
        return sorted(class_checkpoints)[-1]
    except IndexError:
        raise IndexError(f"No files match pattern '{class_name}.*' in directory {search_dir}")


def generate_checkpoint_file(class_name: str, search_dir: str, mode: str = "checkpoint") -> list[str]:
    class_subdir = find_last_checkpoint_dir(class_name, search_dir)
    filtered_checkpoint_files = filter(lambda path: re.fullmatch(f".*{mode}.*", path) is not None,
                                       os.listdir(os.path.join(search_dir, class_subdir)))
    return list(map(lambda basepath: os.path.join(search_dir, class_subdir, basepath),
                    filtered_checkpoint_files))[0]


def slice_image_paths(paths):
    return [i.split('/')[-1].replace('\\','/') for i in paths]


def save_checkpoint(model, optimizer, loss, config):
    timestamp_folder = create_timestamp_folder(config)
    state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict() if optimizer is not None else None}
    Path(f'./checkpoints/embedding_models/{timestamp_folder}').mkdir(parents=True, exist_ok=True)
    filename=f"./checkpoints/embedding_models/{timestamp_folder}/checkpoint.pth.tar"
    draw_loss_curve(
        history=loss, 
        results_path=f"./checkpoints/embedding_models/{timestamp_folder}"
        )
    torch.save(state, filename)

    with open(f"./checkpoints/embedding_models/{timestamp_folder}/config.yaml", "w") as yaml_file:
        yaml.dump(config.__dict__, yaml_file)




def create_timestamp_folder(config):
    """
    Create a folder name based on the current timestamp.
    Returns:
        folder_name (str): The name of the folder, in the format 'YYYY-MM-DD-HH-MM-SS'.
    """
    current_time = time.localtime()
    folder_name = time.strftime('%Y-%m-%d-%H-%M-%S', current_time)
    return f'{config.model}_{config.pipeline}_{folder_name}'    

def draw_loss_curve(history,results_path):

    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.savefig(results_path + '/loss.png')
