
import torch
from dataclasses import dataclass, field
from tools.generate_labels import ModelArguments, DynamicDataTrainingArguments, TrainingArguments
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import RobertaConfig, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers import HfArgumentParser, TrainingArguments
from transformers import EncoderDecoderModel
import dataclasses
import logging
import os
import sys
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
import torch

import numpy as np
import pandas as pd

from src.dataset import FewShotDataset
from src.models import BertForPromptFinetuning, RobertaForPromptFinetuning, resize_token_type_embeddings
from src.trainer import Trainer
from src.funding_task_preprocessor import processors_mapping, num_labels_mapping, output_modes_mapping, compute_metrics_mapping
from src.processors import bound_mapping
from run import DynamicDataTrainingArguments, ModelArguments, DynamicTrainingArguments

from filelock import FileLock
from datetime import datetime

from copy import deepcopy
from tqdm import tqdm
import json

def main():
    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            # Note: the eval dataloader is sequential, so the examples are in order.
            # We average the logits over each sample for using demonstrations.
            predictions = p.predictions
            num_logits = predictions.shape[-1]
            logits = predictions.reshape([eval_dataset.num_sample, -1, num_logits])
            logits = logits.mean(axis=0)
            
            if num_logits == 1:
                preds = np.squeeze(logits)
            else:
                preds = np.argmax(logits, axis=1)

            # Just for sanity, assert label ids are the same.
            label_ids = p.label_ids.reshape([eval_dataset.num_sample, -1])
            label_ids_avg = label_ids.mean(axis=0)
            label_ids_avg = label_ids_avg.astype(p.label_ids.dtype)
            assert (label_ids_avg - label_ids[0]).mean() < 1e-2
            label_ids = label_ids[0]

            return compute_metrics_mapping[task_name](task_name, preds, label_ids)

        return compute_metrics_fn
    
    parser = HfArgumentParser((ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_fn = RobertaForPromptFinetuning
    model_fn = model_fn.from_pretrained(
            pretrained_model_name_or_path = 'roberta-large', 
            config = RobertaConfig.from_json_file("result/partnership-prompt-demo-16-13-roberta-large-27549/config.json") , 
            state_dict = torch.load("result/partnership-prompt-demo-16-13-roberta-large-27549/pytorch_model.bin")
        )

    special_tokens = []
    tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            additional_special_tokens=special_tokens,
            cache_dir= ".",
        )

    print(tokenizer.vocab_size)
    df = pd.read_csv("inference_data.csv")
    device = torch.device('cuda')
    model_fn.label_word_list = torch.LongTensor([0,1])
    model_fn.data_args = data_args
    model_fn.model_args = model_args
    model_fn.return_full_softmax = True
    model_fn.to(device)
    
    for idx, row in df.iterrows():
        text = row.sentence + "Is a collaboration mentioned in the previous sentence?  _ "
        inputs = tokenizer(text, ignore)
        

        encoded_sequence = torch.FloatTensor(inputs['input_ids'])
        encoded_sequence.resize_(1,len(encoded_sequence))

        padded_sequences = tokenizer(text, padding = True)
        attention_mask = torch.FloatTensor(padded_sequences["attention_mask"])
        attention_mask.resize_(1,len(attention_mask))

        mask_positions = []
        tokenized_text = tokenizer.tokenize(text)

        for i in range(len(tokenized_text)):
            if '_' in tokenized_text[i]:
                tokenized_text[i] = '[MASK]'
                mask_positions.append(i)
        mask_positions = torch.FloatTensor(mask_positions)
        mask_positions.resize_(1,len(mask_positions))
        

        zeros,logit = model_fn.forward(
            input_ids = encoded_sequence.to(device).long(), 
            attention_mask = attention_mask.to(device).long(),
            mask_pos = mask_positions.to(device).long(),
            labels = torch.LongTensor([0,1]))
        # top_10 = torch.topk(logits, 10, dim = 1)[1][0]

        # pd.append({"token" : np.argmax(logit), "word" : np.argmax(logit)},ignore_index = True)
        
        print("ARGMAX")
        print(tokenizer.decode([torch.argmax(logit)], skip_special_tokens = True))
        # print("TOKP")
        # top_k = torch.topk(logit, 10, dim = 1)[1][0]
        # for index in top_k:
        #     print(tokenizer.decode([index]))


      
    # print("#########DATA ARGS#############")
    # print(data_args)
    # # ipdb.runcall(FewShotDataset, data_args,tokenizer, "train", True, kwargs = 'foo')
    # train_dataset = FewShotDataset(
    #     data_args, 
    #     tokenizer=tokenizer, 
    #     mode="train", 
    #     use_demo=True)

    # test_dataset = FewShotDataset(
    #         data_args, 
    #         tokenizer=tokenizer, 
    #         mode="test", 
    #         use_demo=True
    #         )

    # print(train_dataset)

    # model_fn.forward(
    #     input_ids = train_dataset.convert_fn().input_ids,
    #     attention_mask = train_dataset.attention_mask,
    #     mask_pos = train_dataset.mask_pos,
    #     labels = train_dataset.labels

    # )
    
    # train_dataset = FewShotDataset(
    #     data_args, 
    #     tokenizer=tokenizer, 
    #     mode="train", 
    #     use_demo=True)
    

    # print(train_dataset.label_word_list)

    # model_fn.label_word_list = torch.tensor(train_dataset.label_word_list).long().cuda()
   

    # trainer = Trainer(
    #         model=model_fn,
    #         args=training_args,
    #         train_dataset=None,
    #         eval_dataset=None,
    #     )



    

    # print("#############TEST############")
    # print(test_dataset)

    # test_datasets = [test_dataset]
       
    # for test_dataset in test_datasets:
    #         trainer.compute_metrics = build_compute_metrics_fn(test_dataset.args.task_name)
    #         print(trainer.compute_metrics)
    #         output = trainer.evaluate(eval_dataset=test_dataset)
    #         test_result = output.metrics

    #         if trainer.is_world_master():
    #             predictions = output.predictions
    #             num_logits = predictions.shape[-1]
    #             logits = predictions.reshape([test_dataset.num_sample, -1, num_logits]).mean(axis=0)
    #             np.save(os.path.join(training_args.save_logit_dir, "{}-{}-{}.npy".format(test_dataset.task_name, training_args.model_id, training_args.array_id)), logits)

    #         test_results.update(test_result)

  
if __name__ == "__main__":
    main()