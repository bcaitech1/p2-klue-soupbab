import pickle as pickle
import os
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, Trainer, TrainingArguments
from load_data import *

import argparse

import wandb


# seed 고정을 위한 함수
def seed_everything(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)  # if use multi-GPU
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  # np.random.seed(seed)
  # random.seed(seed)


# 평가를 위한 metrics function.
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn"s function
  acc = accuracy_score(labels, preds)
  return {
      "accuracy": acc,
  }


def train(args):
  seed_everything(args.seed)
  # load model and tokenizer
  MODEL_NAME = args.model_name
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  # load dataset
  train_dataset = load_data("/opt/ml/input/data/train/train.tsv")
  #dev_dataset = load_data("./dataset/train/dev.tsv")
  train_label = train_dataset["label"].values
  #dev_label = dev_dataset["label"].values
  
  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  #tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  #RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  # setting model hyperparameter
  model_config = AutoConfig.from_pretrained(MODEL_NAME)
  model_config.num_labels = 42
  model_config.hidden_dropout_prob = args.dropout
  model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
  model.to(device)

  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir=f"./results/{MODEL_NAME}",          # output directory
    save_total_limit=args.save_total_limit,              # number of total save model.
    save_strategy=args.save_strategy,         # save strategy.
    save_steps=args.save_steps,                 # model saving step.
    num_train_epochs=args.epochs,              # total number of training epochs
    learning_rate=args.learning_rate,               # learning_rate
    lr_scheduler_type=args.lr_scheduler_type,          # learning_rate_scheduler
    label_smoothing_factor=args.label_smoothing_factor,         # label_smoothing_factor
    per_device_train_batch_size=args.batch_size,  # batch size per device during training
    #per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=args.warmup_steps,                # number of warmup steps for learning rate scheduler
    weight_decay=args.weight_decay,               # strength of weight decay
    logging_dir="./logs",            # directory for storing logs
    logging_steps=args.logging_steps,              # log saving step.
    #evaluation_strategy="steps", # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    #eval_steps = 500,            # evaluation step.
  )
  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    #eval_dataset=RE_dev_dataset,             # evaluation dataset
    # compute_metrics=compute_metrics         # define metrics function
  )
  wandb.config.update(training_args)

  # train model
  trainer.train()


def main(args):
  wandb.init(project="klue", name="xlm-roberta-final", group="xlm-roberta-large", reinit=True)
  train(args)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_name", type=str, default="xlm-roberta-large")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--epochs", type=int, default=20)
  parser.add_argument("--batch_size", type=int, default=64)
  parser.add_argument("--learning_rate", type=float, default=5e-5)
  parser.add_argument("--lr_scheduler_type", type=str, default="cosine_with_restarts")
  parser.add_argument("--warmup_steps", type=int, default=500)
  parser.add_argument("--weight_decay", type=float, default=0.01)
  parser.add_argument("--dropout", type=float, default=0.1)
  parser.add_argument("--label_smoothing_factor", type=float, default=0.5)

  parser.add_argument("--save_total_limit", type=int, default=1)
  parser.add_argument("--save_strategy", type=str, default="steps")
  parser.add_argument("--save_steps", type=int, default=500)
  parser.add_argument("--logging_steps", type=int, default=100)
  args = parser.parse_args()

  main(args)
