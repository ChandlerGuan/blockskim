from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions, QuestionAnsweringModelOutput)


@dataclass
class BaseModelOutputWithPastAndCrossAttentionsWithSkim(BaseModelOutputWithPastAndCrossAttentions):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    all_skim_mask: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class BaseModelOutputWithPoolingAndCrossAttentionsWithSkim(BaseModelOutputWithPoolingAndCrossAttentions):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    all_skim_mask: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class QuestionAnsweringModelOutputWithSkim(QuestionAnsweringModelOutput):
    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    all_skim_mask: Optional[Tuple[torch.FloatTensor]] = None
