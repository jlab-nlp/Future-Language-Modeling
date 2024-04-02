import torch
import math
from torch import nn, Tensor
from torch.nn.parameter import Parameter
from torch.nn import CrossEntropyLoss, init
from torch.nn import functional as F
from copy import deepcopy
from typing import Tuple
from transformers.models.gpt2.modeling_gpt2 import GPT2_START_DOCSTRING, \
    GPT2PreTrainedModel, GPT2Model, PARALLELIZE_DOCSTRING, DEPARALLELIZE_DOCSTRING, \
    get_device_map, assert_device_map, GPT2_INPUTS_DOCSTRING, _TOKENIZER_FOR_DOC, \
    _CHECKPOINT_FOR_DOC, CausalLMOutputWithCrossAttentions, _CONFIG_FOR_DOC, GPT2Block
from transformers.file_utils import add_start_docstrings, \
    add_start_docstrings_to_model_forward, add_code_sample_docstrings
from transformers.models.gpt2 import GPT2LMHeadModel
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.file_utils import ModelOutput
from transformers import BertForPreTraining


class LinearWithCustomBias(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: torch.Tensor, alpha) -> None:
        super(LinearWithCustomBias, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        # Parameter(torch.Tensor(out_features))

        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        # scaling
        new_bias = (bound + bound) * \
                   (bias - torch.min(bias)) / (torch.max(bias) - torch.min(bias)) - bound * torch.ones(bias.size())
        self.bias = Parameter(alpha * new_bias)
        self.bias.requires_grad = False
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # self.reset_parameters()

    # def reset_parameters(self) -> None:
    #
    #     # if self.bias is not None:
    #     #     fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    #     #     bound = 1 / math.sqrt(fan_in)
    #     #     init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


@add_start_docstrings(
    """
    The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    GPT2_START_DOCSTRING,
)
class GPT2LMHeadUNIGRAMModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight", r"lm_head.bias"]

    def __init__(self, config, ngram_bias):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = LinearWithCustomBias(config.n_embd, config.vocab_size, bias=ngram_bias, alpha=1.0)
        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


@add_start_docstrings(
    """
    The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    GPT2_START_DOCSTRING,
)
class GPT2LMHeadUNIGRAMRNNModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight", r"lm_head.bias"]

    def __init__(self, config, unigram_year_tensor):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        # print(unigram_year_tensor.size())
        self.unigram_year_tensor = unigram_year_tensor.transpose(0, 1).unsqueeze(2).cuda()  # 50237, 20, 1
        self.unigram_freq_lstm = torch.nn.LSTM(input_size=1, hidden_size=32,
                                               batch_first=True)
        self.unigram_freq_linear = torch.nn.Linear(32, 1)

        # self.unigram_freq_tensor = torch.nn.Linear(config.vocab_size, config.vocab_size)(
        #     self.unigram_year_tensor.unsqueeze(0)).squeeze()
        # print(self.unigram_freq_tensor.size())
        # self.unigram_freq_tensor = nn.Transformer(d_model=config.vocab_size, nhead=1, num_encoder_layers=1)(self.unigram_year_tensor).squeeze()

        # Problem 1: start year training bias
        # Problem 2: Too large of the LSTM input and output
        # Problem 3: The final sequence does not be trained during training.
        # Problem 4: Do we needs to add a loss for the unigram part?

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        previous_year_index = kwargs.get("previous_year_index", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "previous_year_index": previous_year_index,
            "is_generate": True
        }

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            previous_year_index=None,
            is_generate=False
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)
        first_year_bias = self.unigram_year_tensor.transpose(0, 1)[0].squeeze()
        first_year_index_mask = (previous_year_index == -1)

        unigram_freq_tensor, _ = self.unigram_freq_lstm(self.unigram_year_tensor)
        unigram_freq_tensor = torch.autograd.Variable(
            self.unigram_freq_linear(unigram_freq_tensor).squeeze().transpose(0, 1))
        if not is_generate:
            lm_logits = self.lm_head(hidden_states) + torch.masked_scatter(unigram_freq_tensor[previous_year_index]
                                                                           .cuda(),
                                                                           first_year_index_mask.unsqueeze(1).cuda(),
                                                                           first_year_bias.repeat(first_year_index_mask
                                                                                                  .sum())
                                                                           .cuda()).unsqueeze(1)
        else:
            lm_logits = self.lm_head(hidden_states)
            lm_logits = lm_logits + unigram_freq_tensor[previous_year_index].unsqueeze(1).cuda()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


@add_start_docstrings(
    """
    The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    GPT2_START_DOCSTRING,
)
class GPT2LMHeadUNIGRAMRNNWindowModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight", r"lm_head.bias"]

    def __init__(self, config, unigram_year_tensor, window_size=3):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        # print(unigram_year_tensor.size())
        self.unigram_year_tensor = unigram_year_tensor.transpose(0, 1).unsqueeze(2)  # .cuda()  # 50237, 20, 1
        self.unigram_freq_lstm = torch.nn.LSTM(input_size=1, hidden_size=32,
                                               batch_first=True)
        self.unigram_freq_linear = torch.nn.Linear(32, 1)
        self.window_size = window_size

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        previous_year_index = kwargs.get("previous_year_index", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "previous_year_index": previous_year_index,
            "is_generate": True
        }

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            previous_year_index=None,
            is_generate=False
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)
        lm_bias = self.build_year_input_tensor(previous_year_index).unsqueeze(1)
        lm_logits = self.lm_head(hidden_states) + lm_bias
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMBiasOutputWithCrossAttentions(
            loss=loss,
            bias=lm_bias,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def build_year_input_tensor(self, previous_year_index):
        batch_size = previous_year_index.size(0)
        vocab_size = self.unigram_year_tensor.size(0)
        year_list = [previous_year_index]
        for i in range(1, self.window_size):
            year_list.append(previous_year_index - i)
        year_list.reverse()
        window_sized_year_index_tensor = torch.stack(year_list).transpose(0, 1)
        filter_start_years_mask = window_sized_year_index_tensor[:, self.window_size - 1] < (-1 + self.window_size)
        start_years_biases = self.unigram_year_tensor.transpose(0, 1)[0:self.window_size].squeeze()
        original_year_tensor = self.unigram_year_tensor.squeeze().transpose(0, 1)
        unfiltered_window_year_tensor = original_year_tensor[window_sized_year_index_tensor]
        start_years_in_batch = (window_sized_year_index_tensor[:, self.window_size - 1] + 1)[filter_start_years_mask]
        if self.window_size == 1:
            scatter_values = start_years_biases.repeat(start_years_in_batch.size(), 1)
        else:
            scatter_values = start_years_biases[start_years_in_batch].unsqueeze(1).repeat(1, self.window_size, 1)
        unfiltered_window_year_bias = self.unigram_freq_linear(
            self.unigram_freq_lstm(unfiltered_window_year_tensor.reshape(
                batch_size * vocab_size, self.window_size, 1))[0]).reshape(batch_size, vocab_size, -1)
        filtered_window_year_bias = unfiltered_window_year_bias.transpose(1, 2).masked_scatter(
            filter_start_years_mask.unsqueeze(1).unsqueeze(1)  # .cuda()
            , scatter_values)
        return filtered_window_year_bias[:, -1, :]

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


class GPT2LMHeadUNIGRAMRNNWindowWeightModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight", r"lm_head.bias"]

    def __init__(self, config, unigram_year_tensor, window_size=3):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.config = config
        # self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.weight = Parameter(torch.Tensor(config.vocab_size, config.n_embd))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        # print(unigram_year_tensor.size())
        self.unigram_year_tensor = unigram_year_tensor.transpose(0, 1).unsqueeze(2)  # .cuda()  # 50237, 20, 1
        self.unigram_freq_lstm = torch.nn.LSTM(input_size=1, hidden_size=config.n_embd,
                                               batch_first=True)
        self.unigram_freq_linear = torch.nn.Linear(32, 1)
        self.window_size = window_size
        self.start_tensor = torch.Tensor(self.window_size, config.vocab_size, config.n_embd)  # .cuda()
        nn.init.kaiming_uniform_(self.start_tensor, a=math.sqrt(5))
        self.bias_head = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        previous_year_index = kwargs.get("previous_year_index", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "previous_year_index": previous_year_index,
            "is_generate": True
        }

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            previous_year_index=None,
            is_generate=False
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        lm_bias = self.build_year_input_tensor(previous_year_index)
        new_weight = self.weight.unsqueeze(0) + 0.001 * self.bias_head(lm_bias)  # TODO: ADD A WEIGHT
        lm_logits = hidden_states.matmul(new_weight.transpose(2, 1))
        # lm_logits = F.linear(hidden_states, new_weight).float()
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMBiasOutputWithCrossAttentions(
            loss=loss,
            bias=lm_bias,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def build_year_input_tensor(self, previous_year_index):
        batch_size = previous_year_index.size(0)
        vocab_size = self.unigram_year_tensor.size(0)
        year_list = [previous_year_index]
        for i in range(1, self.window_size):
            year_list.append(previous_year_index - i)
        year_list.reverse()
        window_sized_year_index_tensor = torch.stack(year_list).transpose(0, 1)
        filter_start_years_mask = window_sized_year_index_tensor[:, self.window_size - 1] < (-1 + self.window_size)
        # start_years_biases = self.unigram_year_tensor.transpose(0, 1)[0:self.window_size].squeeze()
        original_year_tensor = self.unigram_year_tensor.squeeze().transpose(0, 1)
        unfiltered_window_year_tensor = original_year_tensor[window_sized_year_index_tensor]  # .cuda()
        start_years_in_batch = (window_sized_year_index_tensor[:, self.window_size - 1] + 1)[filter_start_years_mask]

        year_lstm_out = \
        self.unigram_freq_lstm(unfiltered_window_year_tensor.reshape(batch_size * vocab_size, self.window_size, 1))[0]
        # batch_size * vocab_size, self.window_size, n_embed
        year_lstm_out = year_lstm_out.reshape(batch_size, vocab_size, self.window_size, self.config.n_embd)

        if self.window_size == 1:
            scatter_values = self.start_tensor.repeat(start_years_in_batch.size(0), 1, 1)
        else:
            scatter_values = self.start_tensor[start_years_in_batch].repeat(self.window_size, 1, 1)
        filtered_window_year_bias = year_lstm_out.masked_scatter(
            filter_start_years_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # .cuda()
            , scatter_values)
        return filtered_window_year_bias[:, :, -1, :]

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


class GPT2LMHeadVocabReprRNNWindowWeightModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight", r"lm_head.bias"]

    def __init__(self, config, year_vocab_repr, window_size=3):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.config = config
        # self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.weight = Parameter(torch.Tensor(config.vocab_size, config.n_embd))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        # self.year_vocab_repr = year_vocab_repr
        self.year_vocab_repr_tensor = nn.Parameter(
            torch.stack([torch.stack(year_vocab) for year_vocab in year_vocab_repr])).cuda()
        self.year_vocab_repr_lstm = torch.nn.LSTM(input_size=1024, hidden_size=config.n_embd, batch_first=True)
        self.bias_head = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.window_size = window_size
        self.post_init()

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        previous_year_index = kwargs.get("previous_year_index", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "previous_year_index": previous_year_index,
            "is_generate": True
        }

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            previous_year_index=None,
            is_generate=False
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        lm_bias = self.build_year_input_tensor(previous_year_index)
        new_weight = self.weight.unsqueeze(0) + self.bias_head(lm_bias)
        lm_logits = hidden_states.matmul(new_weight.transpose(2, 1))
        # lm_logits = F.linear(hidden_states, new_weight).float()
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMBiasOutputWithCrossAttentions(
            loss=loss,
            bias=lm_bias,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def build_year_input_tensor(self, previous_year_index):
        batch_size = previous_year_index.size(0)
        vocab_size = self.config.vocab_size
        year_list = [previous_year_index]
        for i in range(1, self.window_size):
            year_list.append(previous_year_index - i)
        year_list.reverse()
        window_sized_year_index_tensor = torch.stack(year_list).transpose(0, 1)
        unfiltered_vocab_year_tensor = self.year_vocab_repr_tensor[window_sized_year_index_tensor].cuda()
        year_vocab_lstm_out = self.year_vocab_repr_lstm(
            unfiltered_vocab_year_tensor.reshape(batch_size * vocab_size, self.window_size, 1024))[0]
        year_vocab_lstm_out = year_vocab_lstm_out.reshape(batch_size, vocab_size, self.window_size, self.config.n_embd)
        return year_vocab_lstm_out[:, :, -1, :]

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


class GPT2LMHeadUNIGRAMVocabReprRNNWindowWeightModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight", r"lm_head.bias"]

    def __init__(self, config, unigram_year_tensor, year_vocab_repr, window_size=3):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.config = config
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        #self.weight = Parameter(torch.Tensor(config.vocab_size, config.n_embd))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        # print(unigram_year_tensor.size())
        self.unigram_year_tensor = unigram_year_tensor.transpose(0, 1).unsqueeze(2)  # .cuda()  # 50237, 20, 1
        self.unigram_freq_lstm = torch.nn.LSTM(input_size=1, hidden_size=config.n_embd,
                                               batch_first=True)
        self.year_vocab_repr_tensor = nn.Parameter(
            torch.stack([torch.stack(year_vocab) for year_vocab in year_vocab_repr]))  # .cuda()
        self.year_vocab_repr_lstm = torch.nn.LSTM(input_size=1024, hidden_size=config.n_embd, batch_first=True)
        self.unigram_freq_linear = torch.nn.Linear(32, 1)
        self.window_size = window_size
        self.start_tensor = torch.Tensor(self.window_size, config.vocab_size, config.n_embd)  # .cuda()
        nn.init.kaiming_uniform_(self.start_tensor, a=math.sqrt(5))
        self.bias_head_1 = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.bias_head_2 = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        previous_year_index = kwargs.get("previous_year_index", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "previous_year_index": previous_year_index,
            "is_generate": True
        }

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            previous_year_index=None,
            is_generate=False
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        lm_bias_1 = self.build_year_input_tensor(previous_year_index)
        lm_bias_2 = self.build_year_input_tensor_by_vocab(previous_year_index)
        new_weight = self.lm_head.weight.unsqueeze(0) + 0.001 * self.bias_head_1(lm_bias_1) + self.bias_head_1(lm_bias_2)
        lm_logits = hidden_states.matmul(new_weight.transpose(2, 1))
        # lm_logits = F.linear(hidden_states, new_weight).float()
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMBiasOutputWithCrossAttentions(
            loss=loss,
            bias=0.001 * lm_bias_1 + lm_bias_2,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def build_year_input_tensor(self, previous_year_index):
        batch_size = previous_year_index.size(0)
        vocab_size = self.unigram_year_tensor.size(0)
        year_list = [previous_year_index]
        for i in range(1, self.window_size):
            year_list.append(previous_year_index - i)
        year_list.reverse()
        window_sized_year_index_tensor = torch.stack(year_list).transpose(0, 1)
        filter_start_years_mask = window_sized_year_index_tensor[:, self.window_size - 1] < (-1 + self.window_size)
        # start_years_biases = self.unigram_year_tensor.transpose(0, 1)[0:self.window_size].squeeze()
        original_year_tensor = self.unigram_year_tensor.squeeze().transpose(0, 1)
        unfiltered_window_year_tensor = original_year_tensor[window_sized_year_index_tensor]  # .cuda()
        start_years_in_batch = (window_sized_year_index_tensor[:, self.window_size - 1] + 1)[filter_start_years_mask]

        year_lstm_out = \
        self.unigram_freq_lstm(unfiltered_window_year_tensor.reshape(batch_size * vocab_size, self.window_size, 1))[0]
        # batch_size * vocab_size, self.window_size, n_embed
        year_lstm_out = year_lstm_out.reshape(batch_size, vocab_size, self.window_size, self.config.n_embd)

        if self.window_size == 1:
            scatter_values = self.start_tensor.repeat(start_years_in_batch.size(0), 1, 1)
        else:
            scatter_values = self.start_tensor[start_years_in_batch].repeat(self.window_size, 1, 1)
        filtered_window_year_bias = year_lstm_out.masked_scatter(
            filter_start_years_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # .cuda()
            , scatter_values)
        return filtered_window_year_bias[:, :, -1, :]

    def build_year_input_tensor_by_vocab(self, previous_year_index):
        batch_size = previous_year_index.size(0)
        vocab_size = self.config.vocab_size
        year_list = [previous_year_index]
        for i in range(1, self.window_size):
            year_list.append(previous_year_index - i)
        year_list.reverse()
        window_sized_year_index_tensor = torch.stack(year_list).transpose(0, 1)
        unfiltered_vocab_year_tensor = self.year_vocab_repr_tensor[window_sized_year_index_tensor]  # .cuda()
        unfiltered_vocab_year_tensor_reshape = unfiltered_vocab_year_tensor.reshape(batch_size * vocab_size,
                                                                                    self.window_size, 1024)
        # zeros = torch.zeros(1, 2*vocab_size, self.config.n_embd, dtype=unfiltered_vocab_year_tensor.dtype,
        #                     device=unfiltered_vocab_year_tensor.device)
        # init.kaiming_uniform_(zeros, a=math.sqrt(5))
        # hx = (zeros, zeros)
        year_vocab_lstm_out = self.year_vocab_repr_lstm(unfiltered_vocab_year_tensor_reshape)[0]
        year_vocab_lstm_out = year_vocab_lstm_out.reshape(batch_size, vocab_size, self.window_size, self.config.n_embd)
        return year_vocab_lstm_out[:, :, -1, :]

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


class GPT2LMHeadVocabReprLinearRNNWindowWeightModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight", r"lm_head.bias"]

    def __init__(self, config, year_vocab_repr, window_size=3):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.config = config
        # self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.weight = Parameter(torch.Tensor(config.vocab_size, config.n_embd))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        # self.year_vocab_repr = year_vocab_repr
        self.E = Parameter(torch.Tensor(config.n_embd, config.n_embd))
        init.kaiming_uniform_(self.E, a=math.sqrt(5))
        self.year_vocab_repr_tensor = nn.Parameter(
            torch.stack([torch.stack(year_vocab) for year_vocab in year_vocab_repr])).cuda() * self.E
        self.year_vocab_repr_linear = torch.nn.Linear(1024, 64)
        self.year_vocab_repr_lstm = torch.nn.LSTM(input_size=64, hidden_size=config.n_embd, batch_first=True)
        self.window_size = window_size
        self.post_init()

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        previous_year_index = kwargs.get("previous_year_index", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "previous_year_index": previous_year_index,
            "is_generate": True
        }

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            previous_year_index=None,
            is_generate=False
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        lm_bias = self.build_year_input_tensor(previous_year_index)
        new_weight = self.weight.unsqueeze(0) + lm_bias
        lm_logits = hidden_states.matmul(new_weight.transpose(2, 1))
        # lm_logits = F.linear(hidden_states, new_weight).float()
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMBiasOutputWithCrossAttentions(
            loss=loss,
            bias=lm_bias,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def build_year_input_tensor(self, previous_year_index):
        batch_size = previous_year_index.size(0)
        vocab_size = self.config.vocab_size
        year_list = [previous_year_index]
        for i in range(1, self.window_size):
            year_list.append(previous_year_index - i)
        year_list.reverse()
        window_sized_year_index_tensor = torch.stack(year_list).transpose(0, 1)
        unfiltered_vocab_year_tensor = self.year_vocab_repr_tensor[window_sized_year_index_tensor].cuda()
        unfiltered_vocab_year_tensor = self.year_vocab_repr_linear(
            unfiltered_vocab_year_tensor.reshape(batch_size * vocab_size, self.window_size, 1024))
        year_vocab_lstm_out = \
        self.year_vocab_repr_lstm(unfiltered_vocab_year_tensor.reshape(batch_size * vocab_size, self.window_size, 64))[
            0]
        year_vocab_lstm_out = year_vocab_lstm_out.reshape(batch_size, vocab_size, self.window_size, self.config.n_embd)
        return year_vocab_lstm_out[:, :, -1, :]

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


class GPT2LMHeadNGRAMRNNWindowModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight", r"lm_head.bias"]

    def __init__(self, config, unigram_year_tensor, window_size=3):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        # print(unigram_year_tensor.size())
        self.unigram_year_tensor = unigram_year_tensor.transpose(0, 1).unsqueeze(2)  # .cuda()  # 50237, 20, 1
        self.unigram_freq_lstm = torch.nn.LSTM(input_size=1, hidden_size=32,
                                               batch_first=True)
        self.unigram_freq_linear = torch.nn.Linear(32, 1)
        self.window_size = window_size

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        previous_year_index = kwargs.get("previous_year_index", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "previous_year_index": previous_year_index,
            "is_generate": True
        }

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            previous_year_index=None,
            is_generate=False
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)
        lm_bias = self.build_year_input_tensor(previous_year_index).unsqueeze(1)
        lm_logits = self.lm_head(hidden_states) + lm_bias
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMBiasOutputWithCrossAttentions(
            loss=loss,
            bias=lm_bias,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def build_year_input_tensor(self, previous_year_index):
        batch_size = previous_year_index.size(0)
        vocab_size = self.unigram_year_tensor.size(0)
        year_list = [previous_year_index]
        for i in range(1, self.window_size):
            year_list.append(previous_year_index - i)
        year_list.reverse()
        window_sized_year_index_tensor = torch.stack(year_list).transpose(0, 1)
        filter_start_years_mask = window_sized_year_index_tensor[:, self.window_size - 1] < (-1 + self.window_size)
        start_years_biases = self.unigram_year_tensor.transpose(0, 1)[0:self.window_size].squeeze()
        original_year_tensor = self.unigram_year_tensor.squeeze().transpose(0, 1)
        unfiltered_window_year_tensor = original_year_tensor[window_sized_year_index_tensor]
        start_years_in_batch = (window_sized_year_index_tensor[:, self.window_size - 1] + 1)[filter_start_years_mask]
        if self.window_size == 1:
            scatter_values = start_years_biases.repeat(start_years_in_batch.size(), 1)
        else:
            scatter_values = start_years_biases[start_years_in_batch].unsqueeze(1).repeat(1, self.window_size, 1)
        unfiltered_window_year_bias = self.unigram_freq_linear(
            self.unigram_freq_lstm(unfiltered_window_year_tensor.reshape(
                batch_size * vocab_size, self.window_size, 1))[0]).reshape(batch_size, vocab_size, -1)
        filtered_window_year_bias = unfiltered_window_year_bias.transpose(1, 2).masked_scatter(
            filter_start_years_mask.unsqueeze(1).unsqueeze(1)  # .cuda()
            , scatter_values)
        return filtered_window_year_bias[:, -1, :]

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


@dataclass
class CausalLMBiasOutputWithCrossAttentions(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Cross attentions weights after the attention softmax, used to compute the weighted average in the
            cross-attention heads.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `torch.FloatTensor` tuples of length `config.n_layers`, with each tuple containing the cached key,
            value states of the self-attention and the cross-attention layers if model is used in encoder-decoder
            setting. Only relevant if `config.is_decoder = True`.

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    bias: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


class GPT2LMHeadVocabReprRNNWindowConcatModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight", r"lm_head.bias"]

    def __init__(self, config, unigram_year_tensor, year_vocab_repr, window_size=3):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        layer_config = deepcopy(config)
        layer_config.hidden_size = 2 * config.hidden_size
        self.transformer_layer = GPT2Block(layer_config, layer_idx=0)
        self.config = config
        # self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.weight = Parameter(torch.Tensor(config.vocab_size, config.n_embd))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.lm_head = torch.nn.Linear(2 * config.n_embd, config.vocab_size, bias=False)
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        # print(unigram_year_tensor.size())
        # self.unigram_year_tensor = unigram_year_tensor.transpose(0, 1).unsqueeze(2).cuda()  # 50237, 20, 1
        # self.unigram_freq_lstm = torch.nn.LSTM(input_size=1, hidden_size=config.n_embd,
        #                                       batch_first=True)
        self.year_vocab_repr_tensor = nn.Parameter(
            torch.stack([torch.stack(year_vocab) for year_vocab in year_vocab_repr])).cuda()
        self.year_vocab_repr_lstm = torch.nn.LSTM(input_size=1024, hidden_size=config.n_embd, batch_first=True)
        # self.unigram_freq_linear = torch.nn.Linear(32, 1)
        self.window_size = window_size
        # self.start_tensor = torch.Tensor(self.window_size, config.vocab_size, config.n_embd) #.cuda()
        # nn.init.kaiming_uniform_(self.start_tensor, a=math.sqrt(5))
        # self.bias_head_1 = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.bias_head_2 = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.temp_bias_linear = nn.Linear(config.n_embd, 1, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        previous_year_index = kwargs.get("previous_year_index", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "previous_year_index": previous_year_index,
            "is_generate": True
        }

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            previous_year_index=None,
            is_generate=False
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        embed_weight = self.transformer.wte.weight

        # lm_bias_1 = self.build_year_input_tensor(previous_year_index)
        lm_bias_2 = self.build_year_input_tensor_by_vocab(previous_year_index)
        batch_size = int(lm_bias_2.size(0))
        if batch_size == 1:
            temp_bias = self.temp_bias_linear(torch.bmm(torch.unsqueeze(torch.transpose(embed_weight, 1, 0), 0)
                                                        .expand(batch_size, -1, -1), lm_bias_2)).squeeze().unsqueeze(0)
        else:
            temp_bias = self.temp_bias_linear(torch.bmm(torch.unsqueeze(torch.transpose(embed_weight, 1, 0), 0)
                                                        .expand(batch_size, -1, -1), lm_bias_2)).squeeze()

        seq_size = int(hidden_states.size(1))
        hidden_states_cat = torch.cat([hidden_states, temp_bias.unsqueeze(1).expand(-1, seq_size, -1)], dim=2)
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0
        hidden_states_cat = self.transformer_layer(hidden_states=hidden_states_cat,
                                                   attention_mask=attention_mask,
                                                   head_mask=head_mask)[0]
        lm_logits = self.lm_head(hidden_states_cat)
        # lm_logits = F.linear(hidden_states, new_weight).float()
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMBiasOutputWithCrossAttentions(
            loss=loss,
            bias=lm_bias_2,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def build_year_input_tensor(self, previous_year_index):
        batch_size = previous_year_index.size(0)
        vocab_size = self.unigram_year_tensor.size(0)
        year_list = [previous_year_index]
        for i in range(1, self.window_size):
            year_list.append(previous_year_index - i)
        year_list.reverse()
        window_sized_year_index_tensor = torch.stack(year_list).transpose(0, 1)
        filter_start_years_mask = window_sized_year_index_tensor[:, self.window_size - 1] < (-1 + self.window_size)
        # start_years_biases = self.unigram_year_tensor.transpose(0, 1)[0:self.window_size].squeeze()
        original_year_tensor = self.unigram_year_tensor.squeeze().transpose(0, 1)
        unfiltered_window_year_tensor = original_year_tensor[window_sized_year_index_tensor].cuda()
        start_years_in_batch = (window_sized_year_index_tensor[:, self.window_size - 1] + 1)[filter_start_years_mask]

        year_lstm_out = \
        self.unigram_freq_lstm(unfiltered_window_year_tensor.reshape(batch_size * vocab_size, self.window_size, 1))[0]
        # batch_size * vocab_size, self.window_size, n_embed
        year_lstm_out = year_lstm_out.reshape(batch_size, vocab_size, self.window_size, self.config.n_embd)

        if self.window_size == 1:
            scatter_values = self.start_tensor.repeat(start_years_in_batch.size(0), 1, 1)
        else:
            scatter_values = self.start_tensor[start_years_in_batch].repeat(self.window_size, 1, 1)
        filtered_window_year_bias = year_lstm_out.masked_scatter(
            filter_start_years_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1).cuda()
            , scatter_values)
        return filtered_window_year_bias[:, :, -1, :]

    def build_year_input_tensor_by_vocab(self, previous_year_index):
        batch_size = previous_year_index.size(0)
        vocab_size = self.config.vocab_size
        year_list = [previous_year_index]
        for i in range(1, self.window_size):
            year_list.append(previous_year_index - i)
        year_list.reverse()
        window_sized_year_index_tensor = torch.stack(year_list).transpose(0, 1)
        unfiltered_vocab_year_tensor = self.year_vocab_repr_tensor[window_sized_year_index_tensor].cuda()
        unfiltered_vocab_year_tensor_reshape = unfiltered_vocab_year_tensor.reshape(batch_size * vocab_size,
                                                                                    self.window_size, 1024)
        # zeros = torch.zeros(1, 2*vocab_size, self.config.n_embd, dtype=unfiltered_vocab_year_tensor.dtype,
        #                     device=unfiltered_vocab_year_tensor.device)
        # init.kaiming_uniform_(zeros, a=math.sqrt(5))
        # hx = (zeros, zeros)
        year_vocab_lstm_out = self.year_vocab_repr_lstm(unfiltered_vocab_year_tensor_reshape)[0]
        year_vocab_lstm_out = year_vocab_lstm_out.reshape(batch_size, vocab_size, self.window_size, self.config.n_embd)
        return year_vocab_lstm_out[:, :, -1, :]

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


class GPT2LMHeadVocabReprRNNWindowAttentionModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight", r"lm_head.bias"]

    def __init__(self, config, unigram_year_tensor, year_vocab_repr, window_size=3):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        #self.transformer_layer = GPT2Block(config, layer_idx=0)
        self.config = config
        # self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.weight = Parameter(torch.Tensor(config.vocab_size, config.n_embd))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        # print(unigram_year_tensor.size())
        # self.unigram_year_tensor = unigram_year_tensor.transpose(0, 1).unsqueeze(2).cuda()  # 50237, 20, 1
        # self.unigram_freq_lstm = torch.nn.LSTM(input_size=1, hidden_size=config.n_embd,
        #                                       batch_first=True)
        self.year_vocab_repr_tensor = nn.Parameter(
            torch.stack([torch.stack(year_vocab) for year_vocab in year_vocab_repr]))  # .cuda()
        self.year_vocab_repr_lstm = torch.nn.LSTM(input_size=1024, hidden_size=config.n_embd, batch_first=True)
        # self.unigram_freq_linear = torch.nn.Linear(32, 1)
        self.window_size = window_size
        # self.start_tensor = torch.Tensor(self.window_size, config.vocab_size, config.n_embd) #.cuda()
        # nn.init.kaiming_uniform_(self.start_tensor, a=math.sqrt(5))
        # self.bias_head_1 = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.bias_head_2 = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.bias_mha = torch.nn.MultiheadAttention(config.n_embd, 1)
        self.temp_bias_linear = nn.Linear(config.n_embd, 1, bias=False)
        self.alpha = nn.Parameter(torch.ones(1))
        # Initialize weights and apply final processing
        self.post_init()

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        previous_year_index = kwargs.get("previous_year_index", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "previous_year_index": previous_year_index,
            "is_generate": True
        }

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            previous_year_index=None,
            is_generate=False
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # lm_bias_1 = self.build_year_input_tensor(previous_year_index)
        lm_bias_2 = self.build_year_input_tensor_by_vocab(previous_year_index)
        batch_size = int(lm_bias_2.size(0))
        # lm_bias_2 batch_size, vocab_size, hidden_size
        mha_hidden_states = self.bias_mha(hidden_states.permute(1, 0, 2), lm_bias_2.permute(1, 0, 2),
                                          lm_bias_2.permute(1, 0, 2))
        hidden_states = hidden_states + self.alpha*mha_hidden_states[0].permute(1, 0, 2)
        #hidden_states = self.transformer_layer(hidden_states=hidden_states)[0]
        lm_logits = self.lm_head(hidden_states)
        # lm_logits = F.linear(hidden_states, new_weight).float()
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMBiasOutputWithCrossAttentions(
            loss=loss,
            bias=lm_bias_2,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def build_year_input_tensor(self, previous_year_index):
        batch_size = previous_year_index.size(0)
        vocab_size = self.unigram_year_tensor.size(0)
        year_list = [previous_year_index]
        for i in range(1, self.window_size):
            year_list.append(previous_year_index - i)
        year_list.reverse()
        window_sized_year_index_tensor = torch.stack(year_list).transpose(0, 1)
        filter_start_years_mask = window_sized_year_index_tensor[:, self.window_size - 1] < (-1 + self.window_size)
        # start_years_biases = self.unigram_year_tensor.transpose(0, 1)[0:self.window_size].squeeze()
        original_year_tensor = self.unigram_year_tensor.squeeze().transpose(0, 1)
        unfiltered_window_year_tensor = original_year_tensor[window_sized_year_index_tensor].cuda()
        start_years_in_batch = (window_sized_year_index_tensor[:, self.window_size - 1] + 1)[filter_start_years_mask]

        year_lstm_out = \
            self.unigram_freq_lstm(unfiltered_window_year_tensor.reshape(batch_size * vocab_size, self.window_size, 1))[
                0]
        # batch_size * vocab_size, self.window_size, n_embed
        year_lstm_out = year_lstm_out.reshape(batch_size, vocab_size, self.window_size, self.config.n_embd)

        if self.window_size == 1:
            scatter_values = self.start_tensor.repeat(start_years_in_batch.size(0), 1, 1)
        else:
            scatter_values = self.start_tensor[start_years_in_batch].repeat(self.window_size, 1, 1)
        filtered_window_year_bias = year_lstm_out.masked_scatter(
            filter_start_years_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1).cuda()
            , scatter_values)
        return filtered_window_year_bias[:, :, -1, :]

    def build_year_input_tensor_by_vocab(self, previous_year_index):
        batch_size = previous_year_index.size(0)
        vocab_size = self.config.vocab_size
        year_list = [previous_year_index]
        for i in range(1, self.window_size):
            year_list.append(previous_year_index - i)
        year_list.reverse()
        window_sized_year_index_tensor = torch.stack(year_list).transpose(0, 1)
        unfiltered_vocab_year_tensor = self.year_vocab_repr_tensor[window_sized_year_index_tensor]  # .cuda()
        unfiltered_vocab_year_tensor_reshape = unfiltered_vocab_year_tensor.reshape(batch_size * vocab_size,
                                                                                    self.window_size, 1024)
        # zeros = torch.zeros(1, 2*vocab_size, self.config.n_embd, dtype=unfiltered_vocab_year_tensor.dtype,
        #                     device=unfiltered_vocab_year_tensor.device)
        # init.kaiming_uniform_(zeros, a=math.sqrt(5))
        # hx = (zeros, zeros)
        year_vocab_lstm_out = self.year_vocab_repr_lstm(unfiltered_vocab_year_tensor_reshape)[0]
        year_vocab_lstm_out = year_vocab_lstm_out.reshape(batch_size, vocab_size, self.window_size, self.config.n_embd)
        return year_vocab_lstm_out[:, :, -1, :]

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


class GPT2LMHeadVocabReprRNNWindowSigmoidAttentionModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight", r"lm_head.bias"]

    def __init__(self, config, unigram_year_tensor, year_vocab_repr, window_size=3):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        # layer_config = deepcopy(config)
        # layer_config.hidden_size = 2 * config.hidden_size
        # self.transformer_layer = GPT2Block(layer_config, layer_idx=0)
        self.config = config
        # self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.weight = Parameter(torch.Tensor(config.vocab_size, config.n_embd))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.tlm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        # print(unigram_year_tensor.size())
        # self.unigram_year_tensor = unigram_year_tensor.transpose(0, 1).unsqueeze(2).cuda()  # 50237, 20, 1
        # self.unigram_freq_lstm = torch.nn.LSTM(input_size=1, hidden_size=config.n_embd,
        #                                       batch_first=True)
        self.year_vocab_repr_tensor = nn.Parameter(
            torch.stack([torch.stack(year_vocab) for year_vocab in year_vocab_repr]))#.cuda()
        self.year_vocab_repr_lstm = torch.nn.LSTM(input_size=1024, hidden_size=config.n_embd, batch_first=True)
        # self.unigram_freq_linear = torch.nn.Linear(32, 1)
        self.window_size = window_size
        # self.start_tensor = torch.Tensor(self.window_size, config.vocab_size, config.n_embd) #.cuda()
        # nn.init.kaiming_uniform_(self.start_tensor, a=math.sqrt(5))
        # self.bias_head_1 = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.bias_head_2 = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.temp_bias_linear = nn.Linear(config.n_embd, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.m1 = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.m2 = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        previous_year_index = kwargs.get("previous_year_index", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "previous_year_index": previous_year_index,
            "is_generate": True
        }

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            previous_year_index=None,
            is_generate=False
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        embed_weight = self.transformer.wte.weight
        pte_weight = self.transformer.wpe.weight

        # lm_bias_1 = self.build_year_input_tensor(previous_year_index)
        lm_bias_2 = self.build_year_input_tensor_by_vocab(previous_year_index)
        batch_size = int(lm_bias_2.size(0))
        if batch_size == 1:
            temp_bias = self.temp_bias_linear(torch.bmm(torch.unsqueeze(torch.transpose(embed_weight, 1, 0), 0)
                                                        .expand(batch_size, -1, -1), lm_bias_2)).squeeze().unsqueeze(0)
        else:
            temp_bias = self.temp_bias_linear(torch.bmm(torch.unsqueeze(torch.transpose(embed_weight, 1, 0), 0)
                                                        .expand(batch_size, -1, -1), lm_bias_2)).squeeze()
        seq_size = int(hidden_states.size(1))
        # hidden_states_cat = torch.cat([hidden_states, temp_bias.unsqueeze(1).expand(-1, seq_size, -1)], dim=2)
        sigmoid_bias = torch.bmm(
            self.sigmoid(torch.bmm(hidden_states,
                                   self.m1(temp_bias).unsqueeze(1).expand(-1, seq_size, -1).permute(0, 2, 1))),
            self.m2(temp_bias).unsqueeze(1).expand(-1, seq_size, -1))
        # AA = sigmoid_bias.view(sigmoid_bias.size()[0], -1)
        # AA -= AA.min(1, keepdim=True)[0]
        # AA /= AA.max(1, keepdim=True)[0]
        # AA = AA.view(batch_size, seq_size, -1)
        hidden_states_add = hidden_states + sigmoid_bias
        # if attention_mask is not None:
        #     if batch_size <= 0:
        #         raise ValueError("batch_size has to be defined and > 0")
        #     attention_mask = attention_mask.view(batch_size, -1)
        #     # We create a 3D attention mask from a 2D tensor mask.
        #     # Sizes are [batch_size, 1, 1, to_seq_length]
        #     # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        #     # this attention mask is more simple than the triangular masking of causal attention
        #     # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        #     attention_mask = attention_mask[:, None, None, :]
        #
        #     # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        #     # masked positions, this operation will create a tensor which is 0.0 for
        #     # positions we want to attend and -10000.0 for masked positions.
        #     # Since we are adding it to the raw scores before the softmax, this is
        #     # effectively the same as removing these entirely.
        #     attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        #     attention_mask = (1.0 - attention_mask) * -10000.0
        # hidden_states_cat = self.transformer_layer(hidden_states=hidden_states_cat,
        #                                            attention_mask=attention_mask,
        #                                            head_mask=head_mask)[0]
        lm_logits = self.tlm_head(hidden_states_add)
        # lm_logits = F.linear(hidden_states, new_weight).float()
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMBiasOutputWithCrossAttentions(
            loss=loss,
            bias=lm_bias_2,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def build_year_input_tensor(self, previous_year_index):
        batch_size = previous_year_index.size(0)
        vocab_size = self.unigram_year_tensor.size(0)
        year_list = [previous_year_index]
        for i in range(1, self.window_size):
            year_list.append(previous_year_index - i)
        year_list.reverse()
        window_sized_year_index_tensor = torch.stack(year_list).transpose(0, 1)
        filter_start_years_mask = window_sized_year_index_tensor[:, self.window_size - 1] < (-1 + self.window_size)
        # start_years_biases = self.unigram_year_tensor.transpose(0, 1)[0:self.window_size].squeeze()
        original_year_tensor = self.unigram_year_tensor.squeeze().transpose(0, 1)
        unfiltered_window_year_tensor = original_year_tensor[window_sized_year_index_tensor].cuda()
        start_years_in_batch = (window_sized_year_index_tensor[:, self.window_size - 1] + 1)[filter_start_years_mask]

        year_lstm_out = \
        self.unigram_freq_lstm(unfiltered_window_year_tensor.reshape(batch_size * vocab_size, self.window_size, 1))[0]
        # batch_size * vocab_size, self.window_size, n_embed
        year_lstm_out = year_lstm_out.reshape(batch_size, vocab_size, self.window_size, self.config.n_embd)

        if self.window_size == 1:
            scatter_values = self.start_tensor.repeat(start_years_in_batch.size(0), 1, 1)
        else:
            scatter_values = self.start_tensor[start_years_in_batch].repeat(self.window_size, 1, 1)
        filtered_window_year_bias = year_lstm_out.masked_scatter(
            filter_start_years_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1).cuda()
            , scatter_values)
        return filtered_window_year_bias[:, :, -1, :]

    def build_year_input_tensor_by_vocab(self, previous_year_index):
        batch_size = previous_year_index.size(0)
        vocab_size = self.config.vocab_size
        year_list = [previous_year_index]
        for i in range(1, self.window_size):
            year_list.append(previous_year_index - i)
        year_list.reverse()
        window_sized_year_index_tensor = torch.stack(year_list).transpose(0, 1)
        unfiltered_vocab_year_tensor = self.year_vocab_repr_tensor[window_sized_year_index_tensor]#.cuda()
        unfiltered_vocab_year_tensor_reshape = unfiltered_vocab_year_tensor.reshape(batch_size * vocab_size,
                                                                                    self.window_size, 1024)
        # zeros = torch.zeros(1, 2*vocab_size, self.config.n_embd, dtype=unfiltered_vocab_year_tensor.dtype,
        #                     device=unfiltered_vocab_year_tensor.device)
        # init.kaiming_uniform_(zeros, a=math.sqrt(5))
        # hx = (zeros, zeros)
        year_vocab_lstm_out = self.year_vocab_repr_lstm(unfiltered_vocab_year_tensor_reshape)[0]
        year_vocab_lstm_out = year_vocab_lstm_out.reshape(batch_size, vocab_size, self.window_size, self.config.n_embd)
        return year_vocab_lstm_out[:, :, -1, :]

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


class GPT2LMHeadVocabReprRNNWindowVectorDotModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight", r"lm_head.bias"]

    def __init__(self, config, unigram_year_tensor, year_vocab_repr, window_size=3):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        # layer_config = deepcopy(config)
        # layer_config.hidden_size = 2 * config.hidden_size
        # self.transformer_layer = GPT2Block(layer_config, layer_idx=0)
        self.config = config
        # self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.weight = Parameter(torch.Tensor(config.vocab_size, config.n_embd))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.tlm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        # print(unigram_year_tensor.size())
        # self.unigram_year_tensor = unigram_year_tensor.transpose(0, 1).unsqueeze(2).cuda()  # 50237, 20, 1
        # self.unigram_freq_lstm = torch.nn.LSTM(input_size=1, hidden_size=config.n_embd,
        #                                       batch_first=True)
        self.year_vocab_repr_tensor = nn.Parameter(
            torch.stack([torch.stack(year_vocab) for year_vocab in year_vocab_repr]))#.cuda()
        self.year_vocab_repr_lstm = torch.nn.LSTM(input_size=1024, hidden_size=config.n_embd, batch_first=True)
        # self.unigram_freq_linear = torch.nn.Linear(32, 1)
        self.window_size = window_size
        # self.start_tensor = torch.Tensor(self.window_size, config.vocab_size, config.n_embd) #.cuda()
        # nn.init.kaiming_uniform_(self.start_tensor, a=math.sqrt(5))
        # self.bias_head_1 = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.bias_head_2 = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.temp_bias_linear = nn.Linear(config.n_embd, 1, bias=False)
        self.dot_vector = torch.Tensor(1, config.n_embd)
        init.kaiming_uniform_(self.dot_vector, a=math.sqrt(5))
        self.dot_vector = nn.Parameter(self.dot_vector.squeeze())
        # Initialize weights and apply final processing
        self.post_init()

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        previous_year_index = kwargs.get("previous_year_index", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "previous_year_index": previous_year_index,
            "is_generate": True
        }

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            previous_year_index=None,
            is_generate=False
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        embed_weight = self.transformer.wte.weight

        # lm_bias_1 = self.build_year_input_tensor(previous_year_index)
        lm_bias_2 = self.build_year_input_tensor_by_vocab(previous_year_index)
        batch_size = int(lm_bias_2.size(0))
        if batch_size == 1:
            temp_bias = self.temp_bias_linear(torch.bmm(torch.unsqueeze(torch.transpose(embed_weight, 1, 0), 0)
                                                        .expand(batch_size, -1, -1), lm_bias_2)).squeeze().unsqueeze(0)
        else:
            temp_bias = self.temp_bias_linear(torch.bmm(torch.unsqueeze(torch.transpose(embed_weight, 1, 0), 0)
                                                        .expand(batch_size, -1, -1), lm_bias_2)).squeeze()
        dot_temp_bias = torch.mul(torch.tensordot(self.dot_vector, temp_bias, dims=([0], [1])).unsqueeze(1), temp_bias)
        seq_size = int(hidden_states.size(1))
        #hidden_states_cat = torch.cat([hidden_states, dot_temp_bias.unsqueeze(1).expand(-1, seq_size, -1)], dim=2)
        hidden_states_add = hidden_states + dot_temp_bias.unsqueeze(1).expand(-1, seq_size, -1)
        # if attention_mask is not None:
        #     if batch_size <= 0:
        #         raise ValueError("batch_size has to be defined and > 0")
        #     attention_mask = attention_mask.view(batch_size, -1)
        #     # We create a 3D attention mask from a 2D tensor mask.
        #     # Sizes are [batch_size, 1, 1, to_seq_length]
        #     # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        #     # this attention mask is more simple than the triangular masking of causal attention
        #     # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        #     attention_mask = attention_mask[:, None, None, :]
        #
        #     # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        #     # masked positions, this operation will create a tensor which is 0.0 for
        #     # positions we want to attend and -10000.0 for masked positions.
        #     # Since we are adding it to the raw scores before the softmax, this is
        #     # effectively the same as removing these entirely.
        #     attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        #     attention_mask = (1.0 - attention_mask) * -10000.0
        # hidden_states_cat = self.transformer_layer(hidden_states=hidden_states_cat,
        #                                            attention_mask=attention_mask,
        #                                            head_mask=head_mask)[0]
        lm_logits = self.tlm_head(hidden_states_add)
        # lm_logits = F.linear(hidden_states, new_weight).float()
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMBiasOutputWithCrossAttentions(
            loss=loss,
            bias=lm_bias_2,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def build_year_input_tensor(self, previous_year_index):
        batch_size = previous_year_index.size(0)
        vocab_size = self.unigram_year_tensor.size(0)
        year_list = [previous_year_index]
        for i in range(1, self.window_size):
            year_list.append(previous_year_index - i)
        year_list.reverse()
        window_sized_year_index_tensor = torch.stack(year_list).transpose(0, 1)
        filter_start_years_mask = window_sized_year_index_tensor[:, self.window_size - 1] < (-1 + self.window_size)
        # start_years_biases = self.unigram_year_tensor.transpose(0, 1)[0:self.window_size].squeeze()
        original_year_tensor = self.unigram_year_tensor.squeeze().transpose(0, 1)
        unfiltered_window_year_tensor = original_year_tensor[window_sized_year_index_tensor].cuda()
        start_years_in_batch = (window_sized_year_index_tensor[:, self.window_size - 1] + 1)[filter_start_years_mask]

        year_lstm_out = \
        self.unigram_freq_lstm(unfiltered_window_year_tensor.reshape(batch_size * vocab_size, self.window_size, 1))[0]
        # batch_size * vocab_size, self.window_size, n_embed
        year_lstm_out = year_lstm_out.reshape(batch_size, vocab_size, self.window_size, self.config.n_embd)

        if self.window_size == 1:
            scatter_values = self.start_tensor.repeat(start_years_in_batch.size(0), 1, 1)
        else:
            scatter_values = self.start_tensor[start_years_in_batch].repeat(self.window_size, 1, 1)
        filtered_window_year_bias = year_lstm_out.masked_scatter(
            filter_start_years_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1).cuda()
            , scatter_values)
        return filtered_window_year_bias[:, :, -1, :]

    def build_year_input_tensor_by_vocab(self, previous_year_index):
        batch_size = previous_year_index.size(0)
        vocab_size = self.config.vocab_size
        year_list = [previous_year_index]
        for i in range(1, self.window_size):
            year_list.append(previous_year_index - i)
        year_list.reverse()
        window_sized_year_index_tensor = torch.stack(year_list).transpose(0, 1)
        unfiltered_vocab_year_tensor = self.year_vocab_repr_tensor[window_sized_year_index_tensor]#.cuda()
        unfiltered_vocab_year_tensor_reshape = unfiltered_vocab_year_tensor.reshape(batch_size * vocab_size,
                                                                                    self.window_size, 1024)
        # zeros = torch.zeros(1, 2*vocab_size, self.config.n_embd, dtype=unfiltered_vocab_year_tensor.dtype,
        #                     device=unfiltered_vocab_year_tensor.device)
        # init.kaiming_uniform_(zeros, a=math.sqrt(5))
        # hx = (zeros, zeros)
        year_vocab_lstm_out = self.year_vocab_repr_lstm(unfiltered_vocab_year_tensor_reshape)[0]
        year_vocab_lstm_out = year_vocab_lstm_out.reshape(batch_size, vocab_size, self.window_size, self.config.n_embd)
        return year_vocab_lstm_out[:, :, -1, :]

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )



def normalization(w):
    return (w - w.min()) / (w.max() - w.min())


def normalization_ranged(w, a, b):
    return (b - a) * normalization(w) + a


def z_score_standardization(w):
    return (w - w.mean()) / w.std()


class GPT2LMHeadVocabReprRNNWindowSigmoidAttentionDotModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight", r"lm_head.bias"]

    def __init__(self, config, unigram_year_tensor, year_vocab_repr, window_size=3):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        # layer_config = deepcopy(config)
        # layer_config.hidden_size = 2 * config.hidden_size
        # self.transformer_layer = GPT2Block(layer_config, layer_idx=0)
        self.config = config
        # self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.weight = Parameter(torch.Tensor(config.vocab_size, config.n_embd))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.tlm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        # print(unigram_year_tensor.size())
        # self.unigram_year_tensor = unigram_year_tensor.transpose(0, 1).unsqueeze(2).cuda()  # 50237, 20, 1
        # self.unigram_freq_lstm = torch.nn.LSTM(input_size=1, hidden_size=config.n_embd,
        #                                       batch_first=True)
        self.year_vocab_repr_tensor = nn.Parameter(
            torch.stack([torch.stack(year_vocab) for year_vocab in year_vocab_repr]))#.cuda()
        self.year_vocab_repr_lstm = torch.nn.LSTM(input_size=1024, hidden_size=config.n_embd, batch_first=True)
        # self.unigram_freq_linear = torch.nn.Linear(32, 1)
        self.window_size = window_size
        # self.start_tensor = torch.Tensor(self.window_size, config.vocab_size, config.n_embd) #.cuda()
        # nn.init.kaiming_uniform_(self.start_tensor, a=math.sqrt(5))
        # self.bias_head_1 = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.bias_head_2 = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.temp_bias_linear = nn.Linear(config.n_embd, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.m0 = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.m1 = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.m2 = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dot_vector = torch.Tensor(1, config.n_embd)
        init.kaiming_uniform_(self.dot_vector, a=math.sqrt(5))
        self.dot_vector = nn.Parameter(self.dot_vector.squeeze())
        self.alpha = 0.0001#nn.Parameter(torch.ones(1))
        # Initialize weights and apply final processing
        self.post_init()

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        previous_year_index = kwargs.get("previous_year_index", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "previous_year_index": previous_year_index,
            "is_generate": True
        }

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            previous_year_index=None,
            is_generate=False
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        embed_weight = self.transformer.wte.weight
        pos_weight = self.transformer.wpe.weight

        # lm_bias_1 = self.build_year_input_tensor(previous_year_index)
        lm_bias_2 = self.build_year_input_tensor_by_vocab(previous_year_index)
        batch_size = int(lm_bias_2.size(0))
        if batch_size == 1:
            temp_bias = self.temp_bias_linear(torch.bmm(torch.unsqueeze(torch.transpose(embed_weight, 1, 0), 0)
                                                        .expand(batch_size, -1, -1), lm_bias_2)).squeeze().unsqueeze(0)
        else:
            temp_bias = self.temp_bias_linear(torch.bmm(torch.unsqueeze(torch.transpose(embed_weight, 1, 0), 0)
                                                        .expand(batch_size, -1, -1), lm_bias_2)).squeeze()

        seq_size = int(hidden_states.size(1))
        beam_size = int(hidden_states.size(0))
        if seq_size != 1:
            sigmoid_pos_bias = torch.bmm(self.sigmoid(torch.bmm(pos_weight[:seq_size].unsqueeze(0).expand(batch_size, -1, -1),
                                                                self.m0(temp_bias).unsqueeze(1).expand(-1, seq_size,
                                                                                                       -1).permute(0, 2,
                                                                                                                   1))),
                                         temp_bias.unsqueeze(1).expand(-1, seq_size, -1))
        else:
            # TODO:needs to be debug during generation
            sigmoid_pos_bias = torch.bmm(self.sigmoid(torch.bmm(pos_weight[position_ids[0]].unsqueeze(0).expand(beam_size, -1, -1),
                                                                self.m0(temp_bias).unsqueeze(1).expand(beam_size, seq_size,
                                                                                                       -1).permute(0, 2,
                                                                                                                   1))),
                                         temp_bias.unsqueeze(1).expand(beam_size, seq_size, -1))

        sigmoid_bias = torch.bmm(
            self.sigmoid(torch.bmm(hidden_states,
                                   self.m1(sigmoid_pos_bias).permute(0, 2, 1))),
            self.m2(sigmoid_pos_bias))
        #nomalized_sigmoid_bias = normalization_ranged(sigmoid_bias, hidden_states.min(), hidden_states.max())
        #dot_sigmoid_bias = torch.mul(torch.tensordot(self.dot_vector, sigmoid_bias, dims=([0], [2])).unsqueeze(2), sigmoid_bias)
        #nomalized_dot_sigmoid_bias = z_score_standardization(dot_sigmoid_bias)
        # hidden_states_cat = torch.cat([hidden_states, temp_bias.unsqueeze(1).expand(-1, seq_size, -1)], dim=2)
        hidden_states_add = hidden_states + self.alpha*sigmoid_bias
        # if attention_mask is not None:
        #     if batch_size <= 0:
        #         raise ValueError("batch_size has to be defined and > 0")
        #     attention_mask = attention_mask.view(batch_size, -1)
        #     # We create a 3D attention mask from a 2D tensor mask.
        #     # Sizes are [batch_size, 1, 1, to_seq_length]
        #     # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        #     # this attention mask is more simple than the triangular masking of causal attention
        #     # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        #     attention_mask = attention_mask[:, None, None, :]
        #
        #     # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        #     # masked positions, this operation will create a tensor which is 0.0 for
        #     # positions we want to attend and -10000.0 for masked positions.
        #     # Since we are adding it to the raw scores before the softmax, this is
        #     # effectively the same as removing these entirely.
        #     attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        #     attention_mask = (1.0 - attention_mask) * -10000.0
        # hidden_states_cat = self.transformer_layer(hidden_states=hidden_states_cat,
        #                                            attention_mask=attention_mask,
        #                                            head_mask=head_mask)[0]
        lm_logits = self.tlm_head(hidden_states_add)
        # lm_logits = F.linear(hidden_states, new_weight).float()
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMBiasOutputWithCrossAttentions(
            loss=loss,
            bias=lm_bias_2,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def build_year_input_tensor(self, previous_year_index):
        batch_size = previous_year_index.size(0)
        vocab_size = self.unigram_year_tensor.size(0)
        year_list = [previous_year_index]
        for i in range(1, self.window_size):
            year_list.append(previous_year_index - i)
        year_list.reverse()
        window_sized_year_index_tensor = torch.stack(year_list).transpose(0, 1)
        filter_start_years_mask = window_sized_year_index_tensor[:, self.window_size - 1] < (-1 + self.window_size)
        # start_years_biases = self.unigram_year_tensor.transpose(0, 1)[0:self.window_size].squeeze()
        original_year_tensor = self.unigram_year_tensor.squeeze().transpose(0, 1)
        unfiltered_window_year_tensor = original_year_tensor[window_sized_year_index_tensor].cuda()
        start_years_in_batch = (window_sized_year_index_tensor[:, self.window_size - 1] + 1)[filter_start_years_mask]

        year_lstm_out = \
        self.unigram_freq_lstm(unfiltered_window_year_tensor.reshape(batch_size * vocab_size, self.window_size, 1))[0]
        # batch_size * vocab_size, self.window_size, n_embed
        year_lstm_out = year_lstm_out.reshape(batch_size, vocab_size, self.window_size, self.config.n_embd)

        if self.window_size == 1:
            scatter_values = self.start_tensor.repeat(start_years_in_batch.size(0), 1, 1)
        else:
            scatter_values = self.start_tensor[start_years_in_batch].repeat(self.window_size, 1, 1)
        filtered_window_year_bias = year_lstm_out.masked_scatter(
            filter_start_years_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1).cuda()
            , scatter_values)
        return filtered_window_year_bias[:, :, -1, :]

    def build_year_input_tensor_by_vocab(self, previous_year_index):
        batch_size = previous_year_index.size(0)
        vocab_size = self.config.vocab_size
        year_list = [previous_year_index]
        for i in range(1, self.window_size):
            year_list.append(previous_year_index - i)
        year_list.reverse()
        window_sized_year_index_tensor = torch.stack(year_list).transpose(0, 1)
        unfiltered_vocab_year_tensor = self.year_vocab_repr_tensor[window_sized_year_index_tensor]#.cuda()
        unfiltered_vocab_year_tensor_reshape = unfiltered_vocab_year_tensor.reshape(batch_size * vocab_size,
                                                                                    self.window_size, 1024)
        # zeros = torch.zeros(1, 2*vocab_size, self.config.n_embd, dtype=unfiltered_vocab_year_tensor.dtype,
        #                     device=unfiltered_vocab_year_tensor.device)
        # init.kaiming_uniform_(zeros, a=math.sqrt(5))
        # hx = (zeros, zeros)
        year_vocab_lstm_out = self.year_vocab_repr_lstm(unfiltered_vocab_year_tensor_reshape)[0]
        year_vocab_lstm_out = year_vocab_lstm_out.reshape(batch_size, vocab_size, self.window_size, self.config.n_embd)
        return year_vocab_lstm_out[:, :, -1, :]

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


class GPT2LMHeadVocabReprRNNWindowSigmoidAttentionPositionModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight", r"lm_head.bias"]

    def __init__(self, config, unigram_year_tensor, year_vocab_repr, window_size=3):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        # layer_config = deepcopy(config)
        # layer_config.hidden_size = 2 * config.hidden_size
        # self.transformer_layer = GPT2Block(layer_config, layer_idx=0)
        self.config = config
        # self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.weight = Parameter(torch.Tensor(config.vocab_size, config.n_embd))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.tlm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        # print(unigram_year_tensor.size())
        # self.unigram_year_tensor = unigram_year_tensor.transpose(0, 1).unsqueeze(2).cuda()  # 50237, 20, 1
        # self.unigram_freq_lstm = torch.nn.LSTM(input_size=1, hidden_size=config.n_embd,
        #                                       batch_first=True)
        self.year_vocab_repr_tensor = nn.Parameter(
            torch.stack([torch.stack(year_vocab) for year_vocab in year_vocab_repr]))#.cuda()
        self.year_vocab_repr_lstm = torch.nn.LSTM(input_size=1024, hidden_size=config.n_embd, batch_first=True)
        # self.unigram_freq_linear = torch.nn.Linear(32, 1)
        self.window_size = window_size
        # self.start_tensor = torch.Tensor(self.window_size, config.vocab_size, config.n_embd) #.cuda()
        # nn.init.kaiming_uniform_(self.start_tensor, a=math.sqrt(5))
        # self.bias_head_1 = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.bias_head_2 = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.temp_bias_linear = nn.Linear(config.n_embd, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.m1 = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.m2 = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dot_vector = torch.Tensor(1, config.n_embd)
        init.kaiming_uniform_(self.dot_vector, a=math.sqrt(5))
        self.dot_vector = nn.Parameter(self.dot_vector.squeeze())
        # Initialize weights and apply final processing
        self.post_init()

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        previous_year_index = kwargs.get("previous_year_index", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "previous_year_index": previous_year_index,
            "is_generate": True
        }

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            previous_year_index=None,
            is_generate=False
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        embed_weight = self.transformer.wte.weight

        # lm_bias_1 = self.build_year_input_tensor(previous_year_index)
        lm_bias_2 = self.build_year_input_tensor_by_vocab(previous_year_index)
        batch_size = int(lm_bias_2.size(0))
        if batch_size == 1:
            temp_bias = self.temp_bias_linear(torch.bmm(torch.unsqueeze(torch.transpose(embed_weight, 1, 0), 0)
                                                        .expand(batch_size, -1, -1), lm_bias_2)).squeeze().unsqueeze(0)
        else:
            temp_bias = self.temp_bias_linear(torch.bmm(torch.unsqueeze(torch.transpose(embed_weight, 1, 0), 0)
                                                        .expand(batch_size, -1, -1), lm_bias_2)).squeeze()
        seq_size = int(hidden_states.size(1))
        sigmoid_bias = torch.bmm(
            self.sigmoid(torch.bmm(hidden_states,
                                   self.m1(temp_bias).unsqueeze(1).expand(-1, seq_size, -1).permute(0, 2, 1))),
            self.m2(temp_bias).unsqueeze(1).expand(-1, seq_size, -1))
        dot_sigmoid_bias = torch.mul(torch.tensordot(self.dot_vector, sigmoid_bias, dims=([0], [2])).unsqueeze(2), sigmoid_bias)
        # hidden_states_cat = torch.cat([hidden_states, temp_bias.unsqueeze(1).expand(-1, seq_size, -1)], dim=2)
        hidden_states_add = hidden_states + dot_sigmoid_bias
        # if attention_mask is not None:
        #     if batch_size <= 0:
        #         raise ValueError("batch_size has to be defined and > 0")
        #     attention_mask = attention_mask.view(batch_size, -1)
        #     # We create a 3D attention mask from a 2D tensor mask.
        #     # Sizes are [batch_size, 1, 1, to_seq_length]
        #     # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        #     # this attention mask is more simple than the triangular masking of causal attention
        #     # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        #     attention_mask = attention_mask[:, None, None, :]
        #
        #     # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        #     # masked positions, this operation will create a tensor which is 0.0 for
        #     # positions we want to attend and -10000.0 for masked positions.
        #     # Since we are adding it to the raw scores before the softmax, this is
        #     # effectively the same as removing these entirely.
        #     attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        #     attention_mask = (1.0 - attention_mask) * -10000.0
        # hidden_states_cat = self.transformer_layer(hidden_states=hidden_states_cat,
        #                                            attention_mask=attention_mask,
        #                                            head_mask=head_mask)[0]
        lm_logits = self.tlm_head(hidden_states_add)
        # lm_logits = F.linear(hidden_states, new_weight).float()
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMBiasOutputWithCrossAttentions(
            loss=loss,
            bias=lm_bias_2,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def build_year_input_tensor(self, previous_year_index):
        batch_size = previous_year_index.size(0)
        vocab_size = self.unigram_year_tensor.size(0)
        year_list = [previous_year_index]
        for i in range(1, self.window_size):
            year_list.append(previous_year_index - i)
        year_list.reverse()
        window_sized_year_index_tensor = torch.stack(year_list).transpose(0, 1)
        filter_start_years_mask = window_sized_year_index_tensor[:, self.window_size - 1] < (-1 + self.window_size)
        # start_years_biases = self.unigram_year_tensor.transpose(0, 1)[0:self.window_size].squeeze()
        original_year_tensor = self.unigram_year_tensor.squeeze().transpose(0, 1)
        unfiltered_window_year_tensor = original_year_tensor[window_sized_year_index_tensor].cuda()
        start_years_in_batch = (window_sized_year_index_tensor[:, self.window_size - 1] + 1)[filter_start_years_mask]

        year_lstm_out = \
        self.unigram_freq_lstm(unfiltered_window_year_tensor.reshape(batch_size * vocab_size, self.window_size, 1))[0]
        # batch_size * vocab_size, self.window_size, n_embed
        year_lstm_out = year_lstm_out.reshape(batch_size, vocab_size, self.window_size, self.config.n_embd)

        if self.window_size == 1:
            scatter_values = self.start_tensor.repeat(start_years_in_batch.size(0), 1, 1)
        else:
            scatter_values = self.start_tensor[start_years_in_batch].repeat(self.window_size, 1, 1)
        filtered_window_year_bias = year_lstm_out.masked_scatter(
            filter_start_years_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1).cuda()
            , scatter_values)
        return filtered_window_year_bias[:, :, -1, :]

    def build_year_input_tensor_by_vocab(self, previous_year_index):
        batch_size = previous_year_index.size(0)
        vocab_size = self.config.vocab_size
        year_list = [previous_year_index]
        for i in range(1, self.window_size):
            year_list.append(previous_year_index - i)
        year_list.reverse()
        window_sized_year_index_tensor = torch.stack(year_list).transpose(0, 1)
        unfiltered_vocab_year_tensor = self.year_vocab_repr_tensor[window_sized_year_index_tensor]#.cuda()
        unfiltered_vocab_year_tensor_reshape = unfiltered_vocab_year_tensor.reshape(batch_size * vocab_size,
                                                                                    self.window_size, 1024)
        # zeros = torch.zeros(1, 2*vocab_size, self.config.n_embd, dtype=unfiltered_vocab_year_tensor.dtype,
        #                     device=unfiltered_vocab_year_tensor.device)
        # init.kaiming_uniform_(zeros, a=math.sqrt(5))
        # hx = (zeros, zeros)
        year_vocab_lstm_out = self.year_vocab_repr_lstm(unfiltered_vocab_year_tensor_reshape)[0]
        year_vocab_lstm_out = year_vocab_lstm_out.reshape(batch_size, vocab_size, self.window_size, self.config.n_embd)
        return year_vocab_lstm_out[:, :, -1, :]

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


class GPT2LMHeadVocabReprRNNWindowVectorDotBiasModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight", r"lm_head.bias"]

    def __init__(self, config, unigram_year_tensor, year_vocab_repr, window_size=3):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        # layer_config = deepcopy(config)
        # layer_config.hidden_size = 2 * config.hidden_size
        # self.transformer_layer = GPT2Block(layer_config, layer_idx=0)
        self.config = config
        # self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.weight = Parameter(torch.Tensor(config.vocab_size, config.n_embd))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        # print(unigram_year_tensor.size())
        # self.unigram_year_tensor = unigram_year_tensor.transpose(0, 1).unsqueeze(2).cuda()  # 50237, 20, 1
        # self.unigram_freq_lstm = torch.nn.LSTM(input_size=1, hidden_size=config.n_embd,
        #                                       batch_first=True)
        self.year_vocab_repr_tensor = nn.Parameter(
            torch.stack([torch.stack(year_vocab) for year_vocab in year_vocab_repr]))#.cuda()
        self.year_vocab_repr_lstm = torch.nn.LSTM(input_size=1024, hidden_size=config.n_embd, batch_first=True)
        # self.unigram_freq_linear = torch.nn.Linear(32, 1)
        self.window_size = window_size
        # self.start_tensor = torch.Tensor(self.window_size, config.vocab_size, config.n_embd) #.cuda()
        # nn.init.kaiming_uniform_(self.start_tensor, a=math.sqrt(5))
        # self.bias_head_1 = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.bias_head_2 = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.temp_bias_linear = nn.Linear(config.n_embd, 1, bias=False)
        self.dot_vector = torch.Tensor(1, config.n_embd)
        init.kaiming_uniform_(self.dot_vector, a=math.sqrt(5))
        self.dot_vector = nn.Parameter(self.dot_vector.squeeze())
        # Initialize weights and apply final processing
        self.post_init()

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        previous_year_index = kwargs.get("previous_year_index", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "previous_year_index": previous_year_index,
            "is_generate": True
        }

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            previous_year_index=None,
            is_generate=False
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        embed_weight = self.transformer.wte.weight

        # lm_bias_1 = self.build_year_input_tensor(previous_year_index)
        lm_bias_2 = self.build_year_input_tensor_by_vocab(previous_year_index)
        # batch_size = int(lm_bias_2.size(0))
        # if batch_size == 1:
        #     temp_bias = self.temp_bias_linear(torch.bmm(torch.unsqueeze(torch.transpose(embed_weight, 1, 0), 0)
        #                                                 .expand(batch_size, -1, -1), lm_bias_2)).squeeze().unsqueeze(0)
        # else:
        #     temp_bias = self.temp_bias_linear(torch.bmm(torch.unsqueeze(torch.transpose(embed_weight, 1, 0), 0)
        #                                                 .expand(batch_size, -1, -1), lm_bias_2)).squeeze()
        # dot_temp_bias = torch.mul(torch.tensordot(self.dot_vector, temp_bias, dims=([0], [1])).unsqueeze(1), temp_bias)
        seq_size = int(hidden_states.size(1))
        #hidden_states_cat = torch.cat([hidden_states, dot_temp_bias.unsqueeze(1).expand(-1, seq_size, -1)], dim=2)
        #hidden_states_add = hidden_states  #+ dot_temp_bias.unsqueeze(1).expand(-1, seq_size, -1)
        # if attention_mask is not None:
        #     if batch_size <= 0:
        #         raise ValueError("batch_size has to be defined and > 0")
        #     attention_mask = attention_mask.view(batch_size, -1)
        #     # We create a 3D attention mask from a 2D tensor mask.
        #     # Sizes are [batch_size, 1, 1, to_seq_length]
        #     # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        #     # this attention mask is more simple than the triangular masking of causal attention
        #     # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        #     attention_mask = attention_mask[:, None, None, :]
        #
        #     # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        #     # masked positions, this operation will create a tensor which is 0.0 for
        #     # positions we want to attend and -10000.0 for masked positions.
        #     # Since we are adding it to the raw scores before the softmax, this is
        #     # effectively the same as removing these entirely.
        #     attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        #     attention_mask = (1.0 - attention_mask) * -10000.0
        # hidden_states_cat = self.transformer_layer(hidden_states=hidden_states_cat,
        #                                            attention_mask=attention_mask,
        #                                            head_mask=head_mask)[0]
        dot_lm_bias = torch.tensordot(self.dot_vector, lm_bias_2, dims=([0], [2])).unsqueeze(1).expand(-1, seq_size, -1)
        lm_logits = self.lm_head(hidden_states) #+ dot_lm_bias
        # lm_logits = F.linear(hidden_states, new_weight).float()
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMBiasOutputWithCrossAttentions(
            loss=loss,
            bias=lm_bias_2,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def build_year_input_tensor(self, previous_year_index):
        batch_size = previous_year_index.size(0)
        vocab_size = self.unigram_year_tensor.size(0)
        year_list = [previous_year_index]
        for i in range(1, self.window_size):
            year_list.append(previous_year_index - i)
        year_list.reverse()
        window_sized_year_index_tensor = torch.stack(year_list).transpose(0, 1)
        filter_start_years_mask = window_sized_year_index_tensor[:, self.window_size - 1] < (-1 + self.window_size)
        # start_years_biases = self.unigram_year_tensor.transpose(0, 1)[0:self.window_size].squeeze()
        original_year_tensor = self.unigram_year_tensor.squeeze().transpose(0, 1)
        unfiltered_window_year_tensor = original_year_tensor[window_sized_year_index_tensor].cuda()
        start_years_in_batch = (window_sized_year_index_tensor[:, self.window_size - 1] + 1)[filter_start_years_mask]

        year_lstm_out = \
        self.unigram_freq_lstm(unfiltered_window_year_tensor.reshape(batch_size * vocab_size, self.window_size, 1))[0]
        # batch_size * vocab_size, self.window_size, n_embed
        year_lstm_out = year_lstm_out.reshape(batch_size, vocab_size, self.window_size, self.config.n_embd)

        if self.window_size == 1:
            scatter_values = self.start_tensor.repeat(start_years_in_batch.size(0), 1, 1)
        else:
            scatter_values = self.start_tensor[start_years_in_batch].repeat(self.window_size, 1, 1)
        filtered_window_year_bias = year_lstm_out.masked_scatter(
            filter_start_years_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1).cuda()
            , scatter_values)
        return filtered_window_year_bias[:, :, -1, :]

    def build_year_input_tensor_by_vocab(self, previous_year_index):
        batch_size = previous_year_index.size(0)
        vocab_size = self.config.vocab_size
        year_list = [previous_year_index]
        for i in range(1, self.window_size):
            year_list.append(previous_year_index - i)
        year_list.reverse()
        window_sized_year_index_tensor = torch.stack(year_list).transpose(0, 1)
        unfiltered_vocab_year_tensor = self.year_vocab_repr_tensor[window_sized_year_index_tensor]#.cuda()
        unfiltered_vocab_year_tensor_reshape = unfiltered_vocab_year_tensor.reshape(batch_size * vocab_size,
                                                                                    self.window_size, 1024)
        # zeros = torch.zeros(1, 2*vocab_size, self.config.n_embd, dtype=unfiltered_vocab_year_tensor.dtype,
        #                     device=unfiltered_vocab_year_tensor.device)
        # init.kaiming_uniform_(zeros, a=math.sqrt(5))
        # hx = (zeros, zeros)
        year_vocab_lstm_out = self.year_vocab_repr_lstm(unfiltered_vocab_year_tensor_reshape)[0]
        year_vocab_lstm_out = year_vocab_lstm_out.reshape(batch_size, vocab_size, self.window_size, self.config.n_embd)
        return year_vocab_lstm_out[:, :, -1, :]

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )
