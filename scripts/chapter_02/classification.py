from sklearn.metrics import accuracy_score, f1_score

from datasets import load_dataset

# load the DistilBERT tokenizer with AutoTokenizer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    PreTrainedTokenizerBase,
    Trainer
)

from transformers.trainer_utils import PredictionOutput
from typing import Dict, Any

import torch
import logging


def compute_metrics(pred: PredictionOutput) -> Dict[str, float]:
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  f1 = f1_score(labels, preds, average="weighted")
  acc = accuracy_score(labels, preds)

  return {"accuracy": acc, "f1": f1}


def tokenize(batch: Dict[str, Any], tokenizer: PreTrainedTokenizerBase) -> Dict[str, Any]:
    return tokenizer(batch["text"], padding=True, truncation=True)


def training() -> Trainer:
    
    emotions = load_dataset("emotion")

    model_ckpt = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    # use the map method to tokenize the whole dataset
    emotions_encoded = emotions.map(lambda batch: tokenize(batch, tokenizer=tokenizer), batched=True, batch_size=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_labels = 6
    batch_size = 64
    
    model = (AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels)).to(device)
    
    logging_steps = len(emotions_encoded["train"]) // batch_size
    model_name = f"{model_ckpt}-finetuned-emotion"

    training_args = TrainingArguments(output_dir=model_name,
                                      num_train_epochs=2,
                                      learning_rate=2e-5,
                                      per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size,
                                      weight_decay=0.01,
                                      eval_strategy="epoch",
                                      disable_tqdm=False,
                                      logging_steps=logging_steps,
                                      push_to_hub=True,
                                      log_level="error")

    trainer = Trainer(model=model, args=training_args, compute_metrics=compute_metrics,
                    train_dataset=emotions_encoded["train"], eval_dataset=emotions_encoded["validation"],
                    tokenizer=tokenizer)

    trainer.train()
    return trainer

if __name__ == "__main__":
    logging.info("Starting routine")
    trainer = training()