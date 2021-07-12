import torch
from src.trainer import Trainer
from tools.generate_labels import ModelArguments, DynamicDataTrainingArguments, TrainingArguments
from src.models import BertForPromptFinetuning, RobertaForPromptFinetuning, resize_token_type_embeddings
from transformers import RobertaConfig
from transformers import HfArgumentParser, TrainingArguments

model_fn = RobertaForPromptFinetuning
model_fn.from_pretrained('roberta-large', config = RobertaConfig.from_json_file("result/partnership-prompt-demo-16-13-roberta-large-27549/config.json") , state_dict = torch.load("result/partnership-prompt-demo-16-13-roberta-large-27549/pytorch_model.bin"))

special_tokens = []

tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        additional_special_tokens=special_tokens,
        cache_dir= ".",
    )

# parser = HfArgumentParser((ModelArguments, DynamicDataTrainingArguments, TrainingArguments))

trainer = Trainer(
        model=model_fn,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
    )
model_fn.predict("inference_data.csv")
# model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# dataset = FewShotDataset(data_args, tokenizer=tokenizer, mode="test", use_demo=True)
# eval_dataset = 
# trainer.evaluate()
# model_fn.evaluate()