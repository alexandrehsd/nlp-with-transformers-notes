from datasets import (
    get_dataset_config_names,
    load_dataset,
    DatasetDict
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from seqeval.metrics import f1_score

import torch
import torch.nn as nn 

from transformers.modeling_outputs import TokenClassifierOutput
from transformers import (
    AutoTokenizer, 
    XLMRobertaConfig,
    AutoConfig,
    TrainingArguments,
    DataCollatorForTokenClassification,
    Trainer
)
from transformers.models.roberta.modeling_roberta import (
    RobertaModel,
    RobertaPreTrainedModel
)

from collections import defaultdict
from typing import List

def get_dataset(langs: List[str]):
    
    fracs = [0.629, 0.229, 0.084, 0.059]  # language distribution of switzerland

    # Return a DatasetDict if a key doesn't exist
    panx_ch = defaultdict(DatasetDict)
    for lang, frac in zip(langs, fracs):
        # Load monolingual corpus
        ds = load_dataset("xtreme", name=f"PAN-X.{lang}")
        
        # Shuffle and downsample each split according to spoken proportion
        for split in ds:
            panx_ch[lang][split] = (
                ds[split]
                .shuffle(seed=0)
                .select(range(int(frac * ds[split].num_rows)))
            )
    
    return panx_ch


class XLMRobertaForTokenClassification(RobertaPreTrainedModel):
    config_class = XLMRobertaConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        # load model body
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        
        # set up token classification head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # load and initialize weights
        self.init_weights()
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        # use model body to get encoder representations
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, **kwargs)
        
        # apply classifier to encoder representation
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)
        
        # calculate losses
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        # return model output object
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
    

def tag_text(text, tags, model, tokenizer):
    # Get tokens with special characters
    tokens = tokenizer(text).tokens()
    
    # Encode the sequence into IDs
    input_ids = xlmr_tokenizer(text, return_tensors="pt").input_ids.to(device)
    
    # Get predictions as distribution over 7 possible classes
    outputs = model(input_ids)[0]
    
    # Take argmax to get most likely class per token
    predictions = torch.argmax(outputs, dim=2)
    
    # Convert to DataFrame
    preds = [tags.names[p] for p in predictions[0].cpu().numpy()]
    return pd.DataFrame([tokens, preds], index=["Tokens", "Tags"])


def tokenize_and_align_labels(examples):
    tokenized_inputs = xlmr_tokenizer(examples["tokens"], 
                                      truncation=True, 
                                      is_split_into_words=True)
    
    labels = []
    for idx, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=idx)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None or word_idx == previous_word_idx:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_idx])
            
            previous_word_idx = word_idx
            
        labels.append(label_ids)
        
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def encode_panx_dataset(corpus):
    return corpus.map(tokenize_and_align_labels, batched=True, remove_columns=['langs', 'ner_tags', 'tokens'])


def align_predictions(predictions, label_ids):
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    labels_list, preds_list = [], []
    
    for batch_idx in range(batch_size):
        example_labels, example_preds = [], []
        for seq_idx, in range(seq_len):
            # ignore label IDs = -100
            if label_ids[batch_idx, seq_idx] != -100:
                example_labels.append(index2tag[label_ids[batch_idx][seq_idx]])
                example_preds.append(index2tag[preds[batch_idx][seq_idx]])
            
        labels_list.append(example_labels)
        preds_list.append(example_preds)
    return preds_list, labels_list
    

def compute_metrics(eval_pred):
    y_pred, y_true = align_predictions(eval_pred.predictions,
                                       eval_pred.label_ids)
    
    return {"f1": f1_score(y_true, y_pred)}


def model_init():
    return XLMRobertaForTokenClassification.from_pretrained(xlmr_model_name, config=xlmr_config).to(device)
 
 
if __name__ == "__main__":
    
    xtreme_subsets = get_dataset_config_names("xtreme")
    print(f"XTREME has {len(xtreme_subsets)} configurations")
    
    langs = ["de", "fr", "it", "en"]
    panx_ch = get_dataset(langs)
    print({lang: [panx_ch[lang]["train"].num_rows] for lang in langs}, 
          index=["Number of training examples"])
    
    tags = panx_ch["de"]["train"].features["ner_tags"].feature

    xlmr_model_name = "xlm-roberta-base"
    xlmr_tokenizer = AutoTokenizer.from_pretrained(xlmr_model_name)
    
    index2tag = {idx: tag for idx, tag in enumerate(tags.names)}
    tag2index = {tag: idx for idx, tag in enumerate(tags.names)}
    
    xlmr_config = AutoConfig.from_pretrained(xlmr_model_name, 
                                             num_labels=tags.num_classes, 
                                             id2label=index2tag, 
                                             label2id=tag2index)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xlmr_model = (
        XLMRobertaForTokenClassification
        .from_pretrained(xlmr_model_name, config=xlmr_config).to(device)
    )
    
    # encode the dataset for finetuning
    panx_de_encoded = encode_panx_dataset(panx_ch["de"])
    
    num_epochs = 3
    batch_size = 24
    logging_steps = len(panx_de_encoded["train"]) // batch_size
    model_name = f"{xlmr_model_name}-finetuned-panx-de"
    training_args = TrainingArguments(
        output_dir=model_name, log_level="error", num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_steps=1e6, weight_decay=0.01, disable_tqdm=False, logging_steps=logging_steps, push_to_hub=True
    )
    
    # define a _data collator_ so we can pad each input sequence to the largest sequence length in a batch
    data_collator = DataCollatorForTokenClassification(xlmr_tokenizer)
    trainer = Trainer(model_init=model_init, args=training_args,
                      data_collator=data_collator, compute_metrics=compute_metrics,
                      train_dataset=panx_de_encoded["train"],
                      eval_dataset=panx_de_encoded["validation"],
                      tokenizer=xlmr_tokenizer
                      )
    
    trainer.train()

    # trainer.push_to_hub(commit_message="Training completed!")