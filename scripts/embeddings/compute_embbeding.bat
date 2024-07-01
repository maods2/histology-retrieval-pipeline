python ./src/main.py --pipeline=generate_stain --config-file=./config/stain_generation.toml



python src/main.py --pipeline=compute_embedding --config-file=config/inference_autoencoder_test_data.toml
python src/main.py --pipeline=compute_embedding --config-file=config/inference_autoencoder_train_data.toml
python src/main.py --pipeline=compute_embedding --config-file=config/inference_swav_test_data.toml
python src/main.py --pipeline=compute_embedding --config-file=config/inference_swav_train_data.toml
python src/main.py --pipeline=compute_embedding --config-file=config/inference_triplet_test_data.toml
python src/main.py --pipeline=compute_embedding --config-file=config/inference_triplet_train_data.toml

python ./src/main.py --pipeline=compute_semantic_attributes --config-file=./config/inference_semantic_attributes_train_data.toml
python ./src/main.py --pipeline=compute_semantic_attributes --config-file=./config/inference_semantic_attributes_test_data.toml

python ./src/main.py --pipeline=concat --config-file=./config/concat_embeddings.toml
