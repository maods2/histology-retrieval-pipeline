DIR="./embedding_encoder/config"

# Bracs Dataset + Efficient Fine Tuning
# Test
python3 ./embedding_encoder/src/main.py \
    --pipeline=compute_semantic_attributes \
    --config-file="${DIR}/bracs_efficient_finetune/inference_semantic_attributes_test_data.toml"
#train
python3 ./embedding_encoder/src/main.py \
    --pipeline=compute_semantic_attributes \
    --config-file="${DIR}/bracs_efficient_finetune/inference_semantic_attributes_train_data.toml"


# # Fiocruz Dataset + Efficient Fine Tuning
# # Test
# python3 ./embedding_encoder/src/main.py \
#     --pipeline=compute_semantic_attributes \
#     --config-file="${DIR}/fiocruz_efficient_finetune/inference_semantic_attributes_test_data.toml"
# #train
# python3 ./embedding_encoder/src/main.py \
#     --pipeline=compute_semantic_attributes \
#     --config-file="${DIR}/fiocruz_efficient_finetune/inference_semantic_attributes_train_data.toml"


# Fiocruz Dataset + Dino Fine Tuning
# Test
# python3 ./embedding_encoder/src/main.py \
#     --pipeline=compute_semantic_attributes \
#     --config-file="${DIR}/fiocruz_dino/inference_semantic_attributes_test_data.toml"
# #train
# python3 ./embedding_encoder/src/main.py \
#     --pipeline=compute_semantic_attributes \
#     --config-file="${DIR}/fiocruz_dino/inference_semantic_attributes_train_data.toml"

