# Future Language Modeling

## Environment setup

- Make sure you install pytorch and corresponding compatible cuda.
- pip3 -r install requirements.txt

## Generate temporal vocabulary word representation

- First use [train_vocab_representation.sh](https://github.com/jlab-nlp/Future-Language-Modeling/blob/main/train_vocab_representation.sh) with each year's paper to obtain the language model for each year
- Then use [generate_year_vocab_representation.py](https://github.com/jlab-nlp/Future-Language-Modeling/blob/main/data_process/generate_year_vocab_representation.py) to generate temporal vocabulary word representation. 
- Here is a [generated temporal vocabulary word representation](https://drive.google.com/file/d/10R8ziuSadVXyUU-0xf8Ds-kANljgutJP/view?usp=sharing), you can download it into project main directory to use.

## Training

- Use [train_future_language_model.sh](https://github.com/jlab-nlp/Future-Language-Modeling/blob/main/train_future_language_model.sh) to train, remember to include the correct model_type and correct save model path. You can also tune other hyperparameters as you want.

There are different model type in the trainer parameter since we tried different models.

1. GPT-2, model_type:gpt2
2. The word frequency model, model_type:gpt2-unigram-rnn-window
3. The contextual model, model_type:gpt2-vocab-repr-rnn-window-weight
4. The $contextual^2$ model, model_type: gpt2-vocab-repr-rnn-window-sigmoid-attention.

We actually tried other ablations models, but did not included into the paper, you can refer to future_language_model_trainer.py to see corresponding models.

# Generation

- Use [generate.sh](https://github.com/jlab-nlp/Future-Language-Modeling/blob/main/generate.sh) to generate, remember to include the correct model_type and correct saved model path.

# Evaluation

We provide several evaluation tools in [evaluation_tools](https://github.com/jlab-nlp/Future-Language-Modeling/tree/main/evaluation_tools).

For the simple perplexity score, use  [train_future_language_model.sh](https://github.com/jlab-nlp/Future-Language-Modeling/blob/main/train_future_language_model.sh)  with only -eval to obtain the loss, and then  use $e^{loss}$ as  to compute the perplexity in the validation set and test set.

For the content perplexity score, use [compute_perplexity.sh](https://github.com/jlab-nlp/Future-Language-Modeling/blob/main/evaluation_tools/compute_perplexity.sh), please include correct saved model path, you can change your test file in the line [360](https://github.com/jlab-nlp/Future-Language-Modeling/blob/main/evaluation_tools/compute_perplexity.py#L360C1-L360C44) of [compute_perplexity.py](https://github.com/jlab-nlp/Future-Language-Modeling/blob/main/evaluation_tools/compute_perplexity.py).

For the content meteor score, use [evaluate.sh](https://github.com/jlab-nlp/Future-Language-Modeling/blob/main/evaluation_tools/evaluate.sh), please include correct generated file path and test file path.





# Cite

```tex
@inproceedings{
li2024future,
title={Future Language Modeling from Temporal Document History},
author={Changmao Li and Jeffrey Flanigan},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=bRLed9prWC}
}
```









