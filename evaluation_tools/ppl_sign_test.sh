#!/bin/bash

modelpaths=() # Add model paths
modeltypes=() # Add model types

# Read the array values with space
for i in "${!modelpaths[@]}"; do
  echo "loading ${modelpaths[i]}...."
  python3 perplexity_single.py --model_type ${modeltypes[i]} \
  --model_name_or_path ${modelpaths[i]} \
  --length 1024 \
  --num_return_sequences 5 \
  --seed 52 \
  --scaling \
  --repetition_penalty 2 \
  --window_size 2 \
  --output_dir "output_ppl"
done
