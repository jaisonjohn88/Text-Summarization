#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 12:33:33 2021

@author: jaisonjohn
"""

from transformers import (BartTokenizer,BartConfig,
                         BartForConditionalGeneration,
                         Trainer, 
                         TrainingArguments,
                         Seq2SeqTrainingArguments,
                         Seq2SeqTrainer,
                         DataCollatorForSeq2Seq)
import torch
from math import inf
import torch.nn.functional as F
import torch.nn as nn

class loadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
class dyneSummarizer():
    def __init__(self,model_id):
        self.model = BartForConditionalGeneration.from_pretrained(model_id)
        self.tokenizer = BartTokenizer.from_pretrained(model_id)
        
    def prepare_data(self,dataset_train,dataset_eval):
        model_inputs = {}
        model_inputs["train"] = self.tokenizer(dataset_train['articles'][0]['text'],\
                                               return_tensors='pt',padding="max_length", truncation=True)
        with self.tokenizer.as_target_tokenizer():
            model_inputs["train"]['labels'] = self.tokenizer(dataset_train['summary'],\
                                                             return_tensors='pt',padding="max_length", truncation=True)\
                                                            ['input_ids']
                                                            
        model_inputs["validation"] = self.tokenizer(dataset_eval['articles'][0]['text'],\
                                                    return_tensors='pt',padding="max_length", truncation=True)
        with self.tokenizer.as_target_tokenizer():
            model_inputs["validation"]['labels'] = self.tokenizer(dataset_eval['summary'],\
                                                                  return_tensors='pt',padding="max_length", truncation=True)\
                                                                ['input_ids']
        train_dataset = loadDataset(model_inputs["train"], model_inputs["train"]['labels'])
        val_dataset = loadDataset(model_inputs["validation"], model_inputs["validation"]['labels'])
        
        return train_dataset, val_dataset
        
    def fine_tune(self,train_dataset,val_dataset):
        training_args = TrainingArguments(
            output_dir='./results',          # output directory
            num_train_epochs=1,              # total number of training epochs
            per_device_train_batch_size=1,  # batch size per device during training
            per_device_eval_batch_size=1,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=10,
        )
        trainModel = self.model
        trainer = Trainer(
            model=trainModel,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset             # evaluation dataset
        )
        
        trainer.train()
        
        return trainModel
        #return trainer
    

class BartMultiDocumentSummariser():
    def __init__(self,model_name,trainedModel = None):
        if trainedModel is None:
            model = BartForConditionalGeneration.from_pretrained(model_name)
        else:
            model = trainedModel
        
        self.encoder = model.model.encoder
        self.decoder = model.model.decoder
        # Construct a linear layer using the final layer and bias of the base model.
        self.linear = nn.Linear(model.model.shared.weight.shape[1], model.model.shared.weight.shape[0])
        self.linear.weight = model.model.shared.weight
        self.linear.bias = nn.Parameter(model.final_logits_bias)
        
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        
    @torch.no_grad()
    def generate(self, encoder_input_ids, decoder_input_ids, num_beams, beam_width, maxlen, no_repeat_ngram_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        encoder_input_ids = encoder_input_ids.to(device)
        decoder_input_ids = decoder_input_ids.to(device)

        # Get the encoder output once.
        encoder_output = self.encoder(encoder_input_ids)
        sequences = [(decoder_input_ids, torch.zeros(1, device=device))]
        for _ in range(maxlen):
            
            all_candidates = list()
            for seq, score in sequences:
                decoder_input_ids = seq.repeat(encoder_input_ids.shape[0], 1)
                dec_output = self.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_output[0],
                )
                output = dec_output[0].mean(dim=0)
                predictions = self.linear(output)
                # Use -inf to ensure that previous ngram will not be selected
                predictions[:, seq[-no_repeat_ngram_size:]] = -inf
                predictions = F.softmax(predictions[-1, :], dim=0)
                probs, idxs = predictions.topk(beam_width)
                for prob, idx in zip(probs, idxs):
                    candidate = torch.cat([seq, idx.view(1, 1)], dim=1)
                    candidate_score = score - prob.log()
                    all_candidates.append((candidate, candidate_score))
            # order all candidates by score
            ordered = sorted(all_candidates, key=lambda tup: tup[1])
            # select k best
            sequences = ordered[:num_beams]
            
            # for arb in ordered:
            #     print('Sequence: ',self.tokenizer.decode(arb[0][0].tolist()))
            #     print('Score: ',arb[1])
            # print('*'*30)
            
            print('Iteration:', _)

            #Stopping criteria when <EOS> is reached
            if 2 in ordered[0][0][0].tolist():
                eos_idx = ordered[0][0][0].tolist().index(2)
                sequences[0][0][0][eos_idx+1:] = 2
                break
                
        return sequences
        