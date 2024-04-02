python3 compute_perplexity.py --model_type gpt \
--model_name_or_path gpt2 \
--length 1024 \
--num_return_sequences 5 \
--seed 52 \
--scaling \
--repetition_penalty 2 \
--window_size 2
