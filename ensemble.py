from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np


def soft_inference(model, tokenized_sent, device):
  dataloader = DataLoader(tokenized_sent, batch_size=40, shuffle=False)
  model.eval()
  output_pred = []
  
  for i, data in enumerate(dataloader):
    with torch.no_grad():
      outputs = model(
          input_ids=data["input_ids"].to(device),
          attention_mask=data["attention_mask"].to(device),
          # token_type_ids=data["token_type_ids"].to(device)          # xlm-roberta는 token_type_ids를 사용하지 않음.
          )
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    
    output_pred.append(logits)

  return np.array(output_pred)


def load_test_dataset(dataset_dir, tokenizer):
  test_dataset = load_data(dataset_dir)
  test_label = test_dataset["label"].values
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer)
  return tokenized_test, test_label


def main():
  """
    주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  models = ["xlm-roberta-large", "monologg/koelectra-base-v3-discriminator", "bert-base-multilingual-cased"]
  checkpoints = [2500, 9000, 2000]
  k_folds = [False, 5, 5]

  # 각 모델마다 one-hot을 안 거친 결과 반환
  predictions = []
  for model_name, checkpoint, k_fold in zip(models, checkpoints, k_folds):
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # load test datset
    test_dataset_dir = "/opt/ml/input/data/test/test.tsv"
    test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    test_dataset = RE_Dataset(test_dataset, test_label)

    if k_fold is not False:
      for i in range(k_fold):
        # load my model
        model = AutoModelForSequenceClassification.from_pretrained(f"./results/kfold_{i}_{model_name}/checkpoint-{str(checkpoint)}")
        model.to(device)

        # predict answer
        pred_answer = soft_inference(model, test_dataset, device)
        predictions.append(pred_answer)

    else:
      # load my model
      model = AutoModelForSequenceClassification.from_pretrained(f"./results/{model_name}/checkpoint-{str(checkpoint)}")
      model.to(device)

      # predict answer
      pred_answer = soft_inference(model, test_dataset, device)
      predictions.append(pred_answer)

  # 모든 모델들의 예측값을 더한 후 one-hot encoding
  predictions = np.array(predictions)
  predictions = predictions.sum(axis=0)
  predictions = np.argmax(predictions, axis=-1)
  predictions = np.array(predictions).flatten()

  # make csv file with predicted answer
  # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
  output = pd.DataFrame(predictions, columns=["pred"])
  output.to_csv(f"./prediction/ensemble.csv", index=False)


if __name__ == "__main__":
  main()
  