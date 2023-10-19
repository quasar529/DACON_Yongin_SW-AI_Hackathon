import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
import os
import random
import sys
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import wandb
import copy
import torch.nn as nn
import math
from tqdm import tqdm, trange
from sklearn.metrics import f1_score
from dataset import ready_data, TweetDataset
from transformers import EarlyStoppingCallback
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    TaskType,
    PeftModel,
    PeftConfig,
)
import loralib as lora
from collections import Counter
import glob
import time

datetime = time.strftime("%Y_%m_%d_%H_%M", time.localtime(time.time()))
'''
Inspired by https://github.com/microsoft/LoRA
'''
def add_lora_to_roberta(model, dim, rank, lora_alpha):
    len_of_layers = len(model.roberta.encoder.layer)  # len(model.roberta.encoder)
    for i in range(len_of_layers):
        model.roberta.encoder.layer[i].attention.self.query = copy.deepcopy(
            lora.Linear(dim, dim, r=rank, lora_alpha=lora_alpha, merge_weights=False, lora_dropout=0.1)
        )
        model.roberta.encoder.layer[i].attention.self.value = copy.deepcopy(
            lora.Linear(dim, dim, r=rank, lora_alpha=lora_alpha, merge_weights=False, lora_dropout=0.1)
        )
        model.roberta.encoder.layer[i].attention.self.key = copy.deepcopy(
            lora.Linear(dim, dim, r=rank, lora_alpha=lora_alpha, merge_weights=False, lora_dropout=0.1)
        )

        model.roberta.encoder.layer[i].attention.output.dense = copy.deepcopy(
            lora.Linear(dim, dim, r=rank, lora_alpha=lora_alpha, merge_weights=False, lora_dropout=0.1)
        )

def add_lora_to_roberta_cl(model, dim, rank, lora_alpha):
    model.classifier.dense = copy.deepcopy(
        lora.Linear(dim, dim, r=rank, lora_alpha=lora_alpha, merge_weights=False, lora_dropout=0.1)
    )
    model.classifier.out_proj = copy.deepcopy(
        lora.Linear(dim, 3, r=rank, lora_alpha=lora_alpha, merge_weights=False, lora_dropout=0.1)
    )

def copy_weights(new_model, W_model):
    """
    W_model의 W weight를 new_model의 W weight로 복사
    """
    len_of_layers = len(new_model.roberta.encoder.layer)
    q_encoder_weight_list = []
    v_encoder_weight_list = []
    q_encoder_bias_list = []
    v_encoder_bias_list = []

    k_encoder_weight_list = []
    k_encoder_bias_list = []

    for i in range(len_of_layers):
        q_encoder_new_weight = W_model.roberta.encoder.layer[i].attention.self.query.weight.data
        q_encoder_weight_list.append(q_encoder_new_weight)
        q_encoder_new_bias = W_model.roberta.encoder.layer[i].attention.self.query.bias.data
        q_encoder_bias_list.append(q_encoder_new_bias)

        v_encoder_new_weight = W_model.roberta.encoder.layer[i].attention.self.value.weight.data
        v_encoder_weight_list.append(v_encoder_new_weight)
        v_encoder_new_bias = W_model.roberta.encoder.layer[i].attention.self.value.bias.data
        v_encoder_bias_list.append(v_encoder_new_bias)

        k_encoder_new_weight = W_model.roberta.encoder.layer[i].attention.self.key.weight.data
        k_encoder_weight_list.append(k_encoder_new_weight)

        k_encoder_new_bias = W_model.roberta.encoder.layer[i].attention.self.key.bias.data
        k_encoder_bias_list.append(k_encoder_new_bias)

    with torch.no_grad():
        for i in range(len_of_layers):
            new_model.roberta.encoder.layer[i].attention.self.query.weight.data.copy_(q_encoder_weight_list[i])
            new_model.roberta.encoder.layer[i].attention.self.value.weight.data.copy_(v_encoder_weight_list[i])
            
            new_model.roberta.encoder.layer[i].attention.self.query.bias.data.copy_(q_encoder_bias_list[i])
            new_model.roberta.encoder.layer[i].attention.self.value.bias.data.copy_(v_encoder_bias_list[i])

            new_model.roberta.encoder.layer[i].attention.self.key.weight.data.copy_(k_encoder_weight_list[i])
            new_model.roberta.encoder.layer[i].attention.self.key.bias.data.copy_(k_encoder_bias_list[i])

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1_macro = f1_score(labels, preds, average="macro")
    print(f"f1_macro : {f1_macro}")
    return {
        "f1": f1_macro,
    }

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train():
    model_name = "roberta-base"
    output_dir = f"./lora8_a32_qkv_ga_{model_name}"

    train_texts, val_texts, train_labels, val_labels, test_texts = ready_data()

    tokenizer = AutoTokenizer.from_pretrained(f"{model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(f"{model_name}", num_labels=3)


    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = TweetDataset(train_encodings, train_labels)
    val_dataset = TweetDataset(val_encodings, val_labels)
    test_dataset = TweetDataset(test_encodings)
    
    print(model)
    new_model = copy.deepcopy(model)
    # model에 LoRA layer 삽입
    add_lora_to_roberta(model, 1024, 8, 32)
    add_lora_to_roberta_cl(model, 1024, 8, 32)
    copy_weights(model, new_model)
    
    trainable_params = []
    trainable_params.append("lora")

    print(count_parameters(model))
    # LoRA 외 다른 weight freeze
    for name, param in model.named_parameters():
        if name.startswith("roberta") or name.startswith("deberta"):
            param.requires_grad = False
            for trainable_param in trainable_params:
                if trainable_param in name:
                    param.requires_grad = True
                    print(f"Trainalble LAYER NAME from LoRA : {name}")
        else:
            print(f"Trainalble LAYER NAME from CL : {name}")
            param.requires_grad = True
            
    print(format(count_parameters(model), ","))
    
    training_args = TrainingArguments(
        output_dir=f"{output_dir}",
        num_train_epochs=60,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_ratio=0.03,
        weight_decay=0.01,
        learning_rate=4e-4,
        fp16=True,
        fp16_opt_level="O2",
        fp16_full_eval=True,
        evaluation_strategy="epoch",
        do_eval=True,
        do_train=True,
        seed=42,
        save_total_limit=2,
        report_to="wandb",
        gradient_accumulation_steps=32,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=10,
        save_strategy="epoch",
        run_name=f"{datetime}_{model_name}_lora8_a32_qkv_cl_ga_32",
        push_to_hub=True,
        label_smoothing_factor=0.2,
        # lr_scheduler_type="cosine_with_restarts",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    trainer.train()
    torch.save(lora.lora_state_dict(model), f"{output_dir}/{model_name}lora8_a32_qkv_ga_.pth")

    # load best model 했는지 확인
    trainer.evaluate()

    preds = trainer.predict(test_dataset).predictions
    predicted_labels = np.argmax(preds, axis=1)
    submit = pd.read_csv("./sample_submission.csv")
    submit["sentiment"] = predicted_labels
    submit.to_csv(f"{output_dir}/{model_name}_lora8_a32_qkv_cl_ga_32.csv", index=False)
    print("Done")

def hard_voting():
    file_paths = glob.glob("/home/lab/bumjun/toy_project/yongin/ensemble_csv/*.csv")
    votes = []

    for file_path in file_paths:
        pred = pd.read_csv(file_path)
        labels = pred["sentiment"].tolist()
        votes.append(labels)

    ensemble_result = []
    for i in range(len(votes[0])):
        # Counter 객체를 이용해 각 클래스별 투표 수 계산 후 가장 많이 나온 클래스 찾기
        vote_result = Counter([votes[j][i] for j in range(len(votes))]).most_common(1)[0][0]
        ensemble_result.append(vote_result)
    submit = pd.read_csv("./sample_submission.csv")
    submit["sentiment"] = ensemble_result
    submit.to_csv("./ensemble_every_single.csv", index=False)

def main():
    train()

if __name__ == "__main__":
    main()
