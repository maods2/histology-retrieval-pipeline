#!/bin/bash
#fiocruz
# python3 -m binary_classifiers.src.train --config_file=binary_classifiers/config/fiocruz_dino/01_hipercel.yaml --model_name=dino --settings_file=fiocruz_dino
# python3 -m binary_classifiers.src.train --config_file=binary_classifiers/config/fiocruz_dino/02_membra.yaml --model_name=dino --settings_file=fiocruz_dino
# python3 -m binary_classifiers.src.train --config_file=binary_classifiers/config/fiocruz_dino/03_sclerosis.yaml --model_name=dino --settings_file=fiocruz_dino
# python3 -m binary_classifiers.src.train --config_file=binary_classifiers/config/fiocruz_dino/04_normal.yaml --model_name=dino --settings_file=fiocruz_dino
# python3 -m binary_classifiers.src.train --config_file=binary_classifiers/config/fiocruz_dino/05_podoc.yaml --model_name=dino --settings_file=fiocruz_dino
bash ./scripts/embeddings/compute_embbedings.sh

python3 -m binary_classifiers.src.train --config_file=binary_classifiers/config/fiocruz_dino/06_cresce.yaml --model_name=dino --settings_file=fiocruz_dino
