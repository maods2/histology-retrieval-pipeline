#!/bin/bash
#fiocruz
python3 -m binary_classifiers.src.train --config_file=binary_classifiers/config/fiocruz/01_hipercel.yaml --model_name=efficientnet --settings_file=fiocruz
python3 -m binary_classifiers.src.train --config_file=binary_classifiers/config/fiocruz/02_membra.yaml --model_name=efficientnet --settings_file=fiocruz
python3 -m binary_classifiers.src.train --config_file=binary_classifiers/config/fiocruz/03_sclerosis.yaml --model_name=efficientnet --settings_file=fiocruz
python3 -m binary_classifiers.src.train --config_file=binary_classifiers/config/fiocruz/04_normal.yaml --model_name=efficientnet --settings_file=fiocruz
python3 -m binary_classifiers.src.train --config_file=binary_classifiers/config/fiocruz/05_podoc.yaml --model_name=efficientnet --settings_file=fiocruz
python3 -m binary_classifiers.src.train --config_file=binary_classifiers/config/fiocruz/06_cresce.yaml --model_name=efficientnet --settings_file=fiocruz

#bracs
python3 -m binary_classifiers.src.train --config_file=binary_classifiers/config/bracs/0_N.yaml --model_name=efficientnet --settings_file=bracs
python3 -m binary_classifiers.src.train --config_file=binary_classifiers/config/bracs/1_PB.yaml --model_name=efficientnet --settings_file=bracs
python3 -m binary_classifiers.src.train --config_file=binary_classifiers/config/bracs/2_UDH.yaml --model_name=efficientnet --settings_file=bracs
python3 -m binary_classifiers.src.train --config_file=binary_classifiers/config/bracs/3_FEA.yaml --model_name=efficientnet --settings_file=bracs
python3 -m binary_classifiers.src.train --config_file=binary_classifiers/config/bracs/4_ADH.yaml --model_name=efficientnet --settings_file=bracs
python3 -m binary_classifiers.src.train --config_file=binary_classifiers/config/bracs/5_DCIS.yaml --model_name=efficientnet --settings_file=bracs
python3 -m binary_classifiers.src.train --config_file=binary_classifiers/config/bracs/6_IC.yaml --model_name=efficientnet --settings_file=bracs