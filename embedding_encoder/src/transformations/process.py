import os
from transformations.randstainna import RandStainNA
import cv2
from pathlib import Path
import re

def get_all_image_files(pathlib_root_folder):
    img_extensions = ['.jpg', '.jpeg', '.png']
    img_regex = re.compile('|'.join(img_extensions), re.IGNORECASE)
    image_files = [f for f in pathlib_root_folder.glob('**/*') if f.is_file() and img_regex.search(f.suffix)]
    return image_files

def generate_new_stain(config):

    # path = Path("C:/Users/Maods/Documents/Development/Mestrado/terumo/apps/terumo-model-binary-glomerulus-hypercellularity/data/binary/binary_Podocytopathy/Podocytopathy/")


    for folder in  config.destination_folders:
        path = Path(config.image_dataset_path + folder)
        img_path_list = get_all_image_files(path)
        save_dir_path_mod = config.destination_path + folder

        for stain in config.stains:
            randstainna = RandStainNA(
                yaml_file = f'./src/transformations/{stain}.yaml',
                std_hyper = 0.0,
                distribution = 'normal',
                probability = 1.0,
                is_train = False
            )

            if not os.path.exists(save_dir_path_mod):
                os.mkdir(save_dir_path_mod)

            
            for img_path in img_path_list:
                img = randstainna(cv2.imread(str(img_path)))
                save_img_path = f'{save_dir_path_mod}/{stain}_{str(img_path).split('\\')[-1]}' 
                print(save_img_path)
                cv2.imwrite(save_img_path,img)

