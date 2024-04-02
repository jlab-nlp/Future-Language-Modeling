export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE=$(pwd)/transformer_cache
python3 vocab_representation_traininer.py --model_name_or_path roberta-large \
    --train_file $1.txt \
    --validation_file $1.txt \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --evaluation_strategy epoch \
    --output_dir saved_model/$1 \
    --line_by_line \
    --max_seq_length 1024 \
    --num_train_epochs 300 \
    --save_strategy epoch \
    --save_total_limit 1