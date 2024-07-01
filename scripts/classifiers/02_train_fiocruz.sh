#!/bin/bash

python3 -m src.train --config_file=binary_classifiers/config/fiocruz/01_hipercel.yaml
python3 -m src.train --config_file=binary_classifiers/config/fiocruz/02_membra.yaml
python3 -m src.train --config_file=binary_classifiers/config/fiocruz/03_sclerosis.yaml
python3 -m src.train --config_file=binary_classifiers/config/fiocruz/04_normal.yaml
python3 -m src.train --config_file=binary_classifiers/config/fiocruz/05_podoc.yaml
python3 -m src.train --config_file=binary_classifiers/config/fiocruz/06_cresce.yaml