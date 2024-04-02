export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE=$(pwd)/transformer_cache
python3 future_language_model_trainer --model_name_or_path gpt2 \
                   --model_type gpt2 \
                   --train_file papers_with_year_start_2003_cut_2000_end_2019.csv \
                   --validation_file papers_with_year_start_2020_cut_2000_end_2020.csv \
                   --per_device_train_batch_size 2 \
                   --per_device_eval_batch_size 2 \
                   --do_train \
                   --do_eval \
                   --evaluation_strategy epoch \
                   --output_dir saved_model/train_2003_2019_valid_2020 \
                   --line_by_line \
                   --max_seq_length 1024 \
                   --num_train_epochs 20 \
                   --save_strategy epoch \
                   --use_log \
                   --overwrite_output_dir \
                   --window_size 2
