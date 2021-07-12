import torch
from src.trainer import Trainer

model = Trainer() 
model.load_state_dict(torch.load("result/partnership-prompt-demo-16-13-roberta-large-27549/pytorch_model.bin"))

model.evaluate()