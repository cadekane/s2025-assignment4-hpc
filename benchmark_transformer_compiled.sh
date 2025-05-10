#!/bin/bash

# Model sizes from §2.1.2
declare -A MODELS
MODELS["small"]="768 3072 12 12"
MODELS["medium"]="1024 4096 24 16"
MODELS["large"]="1280 5120 36 20"
MODELS["xl"]="1600 6400 48 25"
MODELS["2.7B"]="2560 10240 32 32"

for SIZE in "${!MODELS[@]}"; do
    read -r D_MODEL D_FF N_LAYERS N_HEADS <<< "${MODELS[$SIZE]}"

    echo "=== Running VANILLA $SIZE ==="
    python benchmark_transformer_compiled.py --d_model $D_MODEL --d_ff $D_FF --num_layers $N_LAYERS --num_heads $N_HEADS --batch_size 2 --seq_len 256 --backward

    echo "=== Running COMPILED $SIZE ==="
    python benchmark_transformer_compiled.py --d_model $D_MODEL --d_ff $D_FF --num_layers $N_LAYERS --num_heads $N_HEADS --batch_size 2 --seq_len 256 --backward --compile
done
