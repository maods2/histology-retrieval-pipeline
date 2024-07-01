#!/bin/bash

#bracs
python3 -m binary_classifiers.src.train --config_file=binary_classifiers/config/bracs/0_N.yaml --model_name=efficientnet --settings_file=bracs
python3 -m binary_classifiers.src.train --config_file=binary_classifiers/config/bracs/1_PB.yaml --model_name=efficientnet --settings_file=bracs
python3 -m binary_classifiers.src.train --config_file=binary_classifiers/config/bracs/2_UDH.yaml --model_name=efficientnet --settings_file=bracs
python3 -m binary_classifiers.src.train --config_file=binary_classifiers/config/bracs/3_FEA.yaml --model_name=efficientnet --settings_file=bracs
python3 -m binary_classifiers.src.train --config_file=binary_classifiers/config/bracs/4_ADH.yaml --model_name=efficientnet --settings_file=bracs
python3 -m binary_classifiers.src.train --config_file=binary_classifiers/config/bracs/5_DCIS.yaml --model_name=efficientnet --settings_file=bracs
python3 -m binary_classifiers.src.train --config_file=binary_classifiers/config/bracs/6_IC.yaml --model_name=efficientnet --settings_file=bracs