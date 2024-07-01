


# Fiocruz Efficientnet  
# python embedding_encoder/src/main.py --pipeline=compute_embedding --config-file=embedding_encoder/config/fiocruz_efficient/inference_test_data.toml
# python embedding_encoder/src/main.py --pipeline=compute_embedding --config-file=embedding_encoder/config/fiocruz_efficient/inference_train_data.toml


# # Fiocruz Dino 
# python embedding_encoder/src/main.py --pipeline=compute_embedding --config-file=embedding_encoder/config/fiocruz_dino/inference_test_data.toml
# python embedding_encoder/src/main.py --pipeline=compute_embedding --config-file=embedding_encoder/config/fiocruz_dino/inference_train_data.toml


# Fiocruz Efficientnet Finetune

python ./embedding_encoder/src/main.py --pipeline=train --config-file=embedding_encoder/config/fiocruz_efficient_finetune/train_autoencoder_efficientnet.toml
python ./embedding_encoder/src/main.py --pipeline=train --config-file=embedding_encoder/config/fiocruz_efficient_finetune/train_swav_efficientnet.toml
python ./embedding_encoder/src/main.py --pipeline=train --config-file=embedding_encoder/config/fiocruz_efficient_finetune/train_triplet_efficientnet.toml

# Fiocruz Efficientnet Finetune

python ./embedding_encoder/src/main.py --pipeline=train --config-file=embedding_encoder/config/fiocruz_efficient_finetune/train_autoencoder_efficientnet.toml
python ./embedding_encoder/src/main.py --pipeline=train --config-file=embedding_encoder/config/fiocruz_efficient_finetune/train_swav_efficientnet.toml
python ./embedding_encoder/src/main.py --pipeline=train --config-file=embedding_encoder/config/fiocruz_efficient_finetune/train_triplet_efficientnet.toml