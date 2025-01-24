from typing import Optional
from dataclasses import dataclass
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertModel,
    BertOnlyMLMHead,
    SequenceClassifierOutput,
)


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel
from transformers import RobertaModel
from transformers import BertForMaskedLM

import torch
import torch.nn as nn
from llm2vec import LLM2Vec
from torch import Tensor, device, nn

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model
from typing import Any, Dict, List, Optional, Tuple, Union

def batch_to_device(batch, target_device: device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch

class EncodingModel_Stella(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
                "McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp",
            )
        self.encoder = LLM2Vec.from_pretrained(
            "McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp",
            peft_model_name_or_path="McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp-supervised",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
            merge_peft=True,
            pooling_mode="mean",
            max_length=256,
        )
        self.encoder.model = self.initialize_peft(
            self.encoder.model,
        )





        ########
        ##
        self.ema_decay = 0.99
        self.queue_size = 128
        self.moco_lambda = 0.05
        self.moco_temperature = 0.05
        self.hiddensize = 4096
        ##
        self.register_buffer("target_encoder", None)
        self.register_buffer("queue", torch.randn(self.queue_size, self.hiddensize))
        self.queue = F.normalize(self.queue, p=2, dim=1)
        self.register_buffer('queue_labels', torch.zeros(self.queue_size))

    def init_target(self, target=None, target_labels=None):
        self.init_target_net()
        self.init_queue(target, target_labels)

    def init_target_net(self):

        #print('----deepcopy')
        self.target_encoder = deepcopy(self.encoder)

        self.target_encoder.eval()

        for pm in self.target_encoder.parameters():
            pm.requires_grad = False
    
    def init_queue(self, target=None, target_labels=None):

        #print('----init_queue')
        if target is not None:
            # assert target.size() == (self.queue_size, self.config.hidden_size)
            self.queue = target.detach()
            self.queue_labels = target_labels
        else:
            # Todo: device
            self.queue = torch.randn(self.queue_size, self.hiddensize)
            self.queue_labels = torch.zeros(self.queue_size)
        
        
        ##
        self.queue = F.normalize(self.queue, p=2, dim=1)

        self.queue.requires_grad = False
        self.queue_labels.requires_grad = False

    @torch.no_grad()
    def ema_update(self):
        for op, tp in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            tp.data = self.ema_decay * tp.data + (1 - self.ema_decay) * op.data
        #print(" ema_update(self)")

    @torch.no_grad()
    def update_queue(self, key, labels):
        #(key, labels) <=> (target, labels)
        self.queue = torch.cat([key.detach(), self.queue], dim=0)
        self.queue = self.queue[0:self.queue_size]
        self.queue_labels = torch.cat([labels, self.queue_labels], dim=0)
        self.queue_labels = self.queue_labels[0:self.queue_size]
            
    def initialize_peft(
        self,
        model,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_modules: Optional[List[str]] = None,
    ):
        if lora_modules is None and model.config.__class__.__name__ in [
            "LlamaConfig",
            "MistralConfig",
            "GemmaConfig",
            "Qwen2Config",
        ]:
            lora_modules = [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        elif lora_modules is None:
            raise ValueError("lora_modules must be specified for this model.")

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=None,
        )

        model = get_peft_model(model, config)
        print(f"Model's Lora trainable parameters:")
        model.print_trainable_parameters()
        return model

    def forward(self, inputs, is_des = False, is_slow = False): # (b, max_length)
        if is_slow == False:
            # features = self.encoder.tokenize(inputs['input'])
            features = self.encoder.tokenize(inputs)
            features = batch_to_device(features, self.config.device)
            embeddings = self.encoder.forward(features)
            return embeddings
        elif is_slow == True:
            features = self.target_encoder.tokenize(inputs)
            features = batch_to_device(features, self.config.device)
            embeddings = self.target_encoder.forward(features)
            return embeddings
        