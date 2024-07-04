# python3 -m binary_classifiers.src.perform_inference_all --checkpoint_dir="artifacts/fiocruz" --test_dataset_root="data/fiocruz/test_binary" \
#     --device="cuda" --mode="min_loss" > logs/inference_log_min_loss \
#     --best_model_dir="best_model_fiocruz" --overwrite_best_model_dir --settings_file=fiocruz


python3 -m binary_classifiers.src.perform_inference_all --checkpoint_dir="artifacts/bracs" --test_dataset_root="data/bracs/test_binary" \
    --device="cuda" --mode="min_loss" > logs/inference_log_min_loss \
    --best_model_dir="best_model_bracs" --overwrite_best_model_dir --settings_file=bracs


# python3 -m binary_classifiers.src.perform_inference_all --checkpoint_dir="artifacts/fiocruz_dino" --test_dataset_root="data/fiocruz_dino/test_binary" \
#     --device="cuda" --mode="min_loss" > logs/inference_log_min_loss \
#     --best_model_dir="best_model_fiocruz_dino" --overwrite_best_model_dir --settings_file=fiocruz_dino