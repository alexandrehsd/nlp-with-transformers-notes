from datasets import load_dataset
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import torch
from transformers import pipeline, set_seed
from transformers import DataCollatorForSeq2Seq
from transformers import TrainingArguments, Trainer
from datasets import load_metric
import pandas as pd


set_seed(42)


def three_sentence_summary(text):
    # baseline model
    return "\n".join(sent_tokenize(text)[:3])


def evaluate_summaries_baseline(dataset, metric, column_text="article", column_summary="highlights"):
    summaries = [three_sentence_summary(text) for text in dataset[column_text]]
    metric.add_batch(predictions=summaries, references=dataset[column_summary])
    score = metric.compute()
    return score


def chunks(list_of_elements, batch_size):
    """Yield successive batch-sized chunks from list_of_elements."""
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i : i + batch_size]
        
        
def evaluate_summaries_bart(dataset, metric, model, tokenizer, batch_size=16, device=device, column_text="article", column_summary="highlights"):
    article_batches = list(chunks(dataset[column_text], batch_size))
    target_batches = list(chunks(dataset[column_summary], batch_size))
    
    for article_batch, target_batch in tqdm(zip(article_batches, target_batches), total=len(article_batches)):
        inputs = tokenizer(article_batch, max_length=1024, truncation=True, padding="max_length", return_tensors="pt")
        
        summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                                   attention_mask=inputs["attention_mask"].to(device),
                                   length_penalty=0.8, num_beams=8, max_length=128)
        decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in summaries]
        
        decoded_summaries = ["\n".join(sent_tokenize(d)) for d in decoded_summaries]
        metric.add_batch(predictions=decoded_summaries, references=target_batch)
        
    score = metric.compute()
    return score


def convert_examples_to_features(example_batch):
    input_encodings = tokenizer(example_batch["dialogue"], max_length=1024, truncation=True)
    
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(example_batch["summary"], max_length=128, truncation=True)
        
    return {"input_ids": input_encodings["input_ids"],
            "attention_mask": input_encodings["attention_mask"],
            "labels": target_encodings["input_ids"]}
    


if __name__ == "__main__":

    nltk.download("punkt")
    dataset_samsum = load_dataset("samsum")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_ckpt = "facebook/bart-base"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)
    rouge_metric = load_metric("rouge")
    
    # evaluating baseline on samsum
    test_sampled = dataset_samsum["test"].shuffle(seed=42)
    score = evaluate_summaries_baseline(test_sampled, rouge_metric, column_text="dialogue", column_summary="summary")

    rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    rouge_dict = dict((rn, score[rn].mid.fmeasure) for rn in rouge_names)
    print(pd.DataFrame.from_dict(rouge_dict, orient="index", columns=["baseline"]).T)
    
    # evaluating bart on samsum
    pipe = pipeline("summarization", model="facebook/bart-base")

    pipe_out = pipe(dataset_samsum["test"][0]["dialogue"])
    print("Summary:")
    print("\n".join(sent_tokenize(pipe_out[0]["summary_text"])))
        
    # fine-tuning bart
    dataset_samsum_pt = dataset_samsum.map(convert_examples_to_features, batched=True)
    columns = ["input_ids", "labels", "attention_mask"]
    dataset_samsum_pt.set_format(type="torch", columns=columns)
    
    seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    training_args = TrainingArguments(
        output_dir='bart-samsum', 
        num_train_epochs=1, 
        warmup_steps=500,
        per_device_train_batch_size=1, 
        per_device_eval_batch_size=1,
        weight_decay=0.01, 
        logging_steps=10, 
        push_to_hub=False,
        evaluation_strategy='steps', 
        eval_steps=500, 
        save_steps=1e6,
        gradient_accumulation_steps=16)
    
    trainer = Trainer(model=model, 
                      args=training_args, 
                      tokenizer=tokenizer, 
                      data_collator=seq2seq_data_collator, 
                      train_dataset=dataset_samsum_pt["train"], 
                      eval_dataset=dataset_samsum_pt["validation"])
    
    trainer.train()
    
    score = evaluate_summaries_bart(dataset_samsum["test"], 
                                    rouge_metric, 
                                    trainer.model, 
                                    tokenizer, 
                                    batch_size=2, 
                                    column_text="dialogue", 
                                    column_summary="summary")
    
    rouge_dict = dict((rn, score[rn].mid.fmeasure) for rn in rouge_names)
    print(pd.DataFrame(rouge_dict, index=[f"bart-samsum"]))