#!/bin/bash
#fiocruz
python3 -m binary_classifiers.src.train --config_file=binary_classifiers/config/fiocruz/01_hipercel.yaml --model_name=dino --settings_file=fiocruz_dino
python3 -m binary_classifiers.src.train --config_file=binary_classifiers/config/fiocruz/02_membra.yaml --model_name=dino --settings_file=fiocruz_dino
python3 -m binary_classifiers.src.train --config_file=binary_classifiers/config/fiocruz/03_sclerosis.yaml --model_name=dino --settings_file=fiocruz_dino
python3 -m binary_classifiers.src.train --config_file=binary_classifiers/config/fiocruz/04_normal.yaml --model_name=dino --settings_file=fiocruz_dino
python3 -m binary_classifiers.src.train --config_file=binary_classifiers/config/fiocruz/05_podoc.yaml --model_name=dino --settings_file=fiocruz_dino
python3 -m binary_classifiers.src.train --config_file=binary_classifiers/config/fiocruz/06_cresce.yaml --model_name=dino --settings_file=fiocruz_dino