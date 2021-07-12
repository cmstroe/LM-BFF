import torch
from src.trainer import Trainer
from src.models import BertForPromptFinetuning, RobertaForPromptFinetuning, resize_token_type_embeddings
from transformers import RobertaConfig

model_fn = RobertaForPromptFinetuning
model_fn.from_pretrained('roberta-large', config = RobertaConfig.from_json_file("result/partnership-prompt-demo-16-13-roberta-large-27549/config.json") , state_dict = torch.load("result/partnership-prompt-demo-16-13-roberta-large-27549/pytorch_model.bin"))

model.evaluate()