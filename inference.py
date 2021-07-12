import torch
from src.trainer import Trainer
from src.models import BertForPromptFinetuning, RobertaForPromptFinetuning, resize_token_type_embeddings

model_fn = RobertaForPromptFinetuning
model.from_pretrained(config = "result/partnership-prompt-demo-16-13-roberta-large-27549/config.json" , state_dict = "result/partnership-prompt-demo-16-13-roberta-large-27549/pytorch_model.bin")

model.evaluate()