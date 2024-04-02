export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE=$(pwd)/transformer_cache
python3 generator.py \
    --model_type=gpt2 \
    --model_name_or_path=gpt2