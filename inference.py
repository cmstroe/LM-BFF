
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
            config = RobertaConfig.from_json_file("results_funding_150/config.json") , 
            state_dict = torch.load("results_funding_150/pytorch_model.bin")
        )
    special_tokens = []
    tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            additional_special_tokens=special_tokens, truncation=True,
            cache_dir= ".",
        )
    df = pd.read_csv("inference_data.csv")
    device = torch.device('cuda:0')
    model_fn.label_word_list = torch.LongTensor([0,1])
    model_fn.data_args = data_args
    model_fn.model_args = model_args
    model_fn.return_full_softmax = True
    model_fn.to(device)
    df_results = pd.DataFrame(columns = ['sentence', "token_values" , "word"])

    for idx, row in df.iterrows():
        text = row.sentence + "Is a investment mentioned in the previous sentence?  _ "
        text = row.sentence + "Is a funding round mentioned in the previous sentence?  _ "
        if len(text) > 512:
            text = text[:512]
        inputs = tokenizer(text)
        
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
        
        with torch.no_grad():
            zeros,logit = model_fn.forward(
            input_ids = encoded_sequence.to('cuda:0').long(), 
            attention_mask = attention_mask.to('cuda:0').long(),
            mask_pos = mask_positions.to('cuda:0').long(),
            labels = torch.LongTensor([0,1]))
        
        try :
            df_results = df_results.append({"sentence" : row.sentence ,
                    "token_values" : torch.topk(logit, 1) ,
                    "word" : tokenizer.decode([torch.argmax(logit)]) if torch.argmax(logit) else "nothing", 
                    },
                    ignore_index = True)
        except:
            print("error")
        del encoded_sequence
        del attention_mask
        del mask_positions
        torch.cuda.empty_cache()
        print(idx)
        # except:
        #     print("invalid character")
        
        torch.cuda.empty_cache()
            
    
    df_results.groupby(['word'])
    df_results.to_csv("df_funding.csv", index = False)
      
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