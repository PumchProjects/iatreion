uv run iatreion --config configs/config.toml process

for model_name in xgboost random-forest rrl; do
    uv run iatreion --config configs/config.toml train $model_name
    uv run iatreion --config configs/config.toml train $model_name -f
    uv run iatreion --config configs/config.toml train result-replay --source-model $model_name
    uv run iatreion --config configs/config.toml train result-replay --source-model $model_name --eval-names h-demo h-mmse h-moca h-mri h-history sh-apoe-labdata
    uv run iatreion --config configs/config.toml train result-replay --source-model $model_name --eval-names h-demo h-mmse h-moca h-mri h-history sh-apoe-labdata -f
    uv run iatreion --config configs/config.toml train result-replay --source-model $model_name --eval-names h-demo h-mmse h-moca h-mri-roi h-history sh-apoe-labdata
    uv run iatreion --config configs/config.toml train result-replay --source-model $model_name --eval-names h-demo h-mmse h-moca h-mri-roi h-history sh-apoe-labdata -f
    uv run iatreion --config configs/config.toml eval $model_name -n h-demo h-mmse h-moca h-mri h-history sh-apoe-labdata
    uv run iatreion --config configs/config.toml eval $model_name -n h-demo h-mmse h-moca h-mri-roi h-history sh-apoe-labdata
done

uv run iatreion --config configs/config.toml train rrl-parser
