import os
import sys
import time
import json

from tqdm import tqdm
from collections import OrderedDict
import numpy as np

import torch

import torchmetrics

class Evaluator:
    def __init__(self, metrics=None):
        if not metrics:
            metrics = ["rouge", "sacre_bleu", "bertscore", "factkb"]
        self.metrics = metrics
    
    def evaluate(self, predictions, references, documents, metrics=["rouge", "bertscore", "factkb"]):
        result_dict = OrderedDict()
        if "rouge" in metrics:
            rouge_dict = self.calculate_rouge(predictions, references)
            for k, v in rouge_dict.items():
                result_dict[k] = v
        if "sacre_bleu" in metrics:
            sacre_bleu_dict = self.calculate_sacrebleu(predictions, references)
            for k, v in sacre_bleu_dict.items():
                result_dict[k] = v
        if "bertscore" in metrics:
            bertscore_dict = self.calculate_bertscore(predictions, references)
            for k, v in bertscore_dict.items():
                result_dict[k] = v
        if "factkb" in metrics:
            result_dict["factkb"] = self.calculate_factkb(predictions, documents)

        for k, v in result_dict.items():
            print(f"{k} -> {v*100:.2f}")
        return result_dict

    def calculate_rouge(self, predictions, references):
        from torchmetrics.functional.text.rouge import rouge_score
        rouge_dict = rouge_score(preds=predictions, target=references)
        return {k: v.item() for k, v in rouge_dict.items()}

    def calculate_sacrebleu(self, predictions, references):
        from torchmetrics.functional.text import sacre_bleu_score
        score = sacre_bleu_score(preds=predictions, target=[[i] for i in references])
        return {"sacre_bleu": score.item()}

    def calculate_bertscore(self, predictions, references):
        import evaluate
        # bertscore = evaluate.load("bertscore")
        bertscore = evaluate.load("/data/disk0/jiawei/na/context-aware-decoding-qfs/src/evaluate/metrics/bertscore/bertscore.py")
        # bertscore = evaluate.load("/data/nayu/CrossSum/lei/transformers/src/metrics/bertscore/")
        # bertscore = evaluate.load("/home/weikaiwen/na/context-aware-decoding-qfs/src/bertscore/")
        bertscore_dict = bertscore.compute(predictions=predictions, references=references, model_type="roberta-large")  # 这里我把model_type在程序里面写死了的
        res = {"bertscore_precision": np.mean(bertscore_dict["precision"]), "bertscore_recall": np.mean(bertscore_dict["recall"]), "bertscore_f1": np.mean(bertscore_dict["f1"])}
        return {k: v.item() for k, v in res.items()}

    def calculate_factkb(self, predictions, documents):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        from transformers import AutoTokenizer
        from transformers import AutoModelForSequenceClassification
        #tokenizer = AutoTokenizer.from_pretrained("/data/disk0/jiawei/na/context-aware-decoding-qfs/data/pretrained_models/roberta-base") # bunsenfeng/factkb
        #model = AutoModelForSequenceClassification.from_pretrained("/data/disk0/jiawei/na/context-aware-decoding-qfs/data/pretrained_models/FactKB", torch_dtype=torch.float16) # bunsenfeng/factkb
        tokenizer = AutoTokenizer.from_pretrained("../data/pretrained_models/roberta-base") # bunsenfeng/factkb
        model = AutoModelForSequenceClassification.from_pretrained("../data/pretrained_models/FactKB", torch_dtype=torch.float16) # bunsenfeng/factkb
        model = model.to(device)
        res = []
        for i in range(len(predictions)):
            input_pretokenized = f"{predictions[i]} {tokenizer.sep_token} {documents[i]}"
            tokenized_input = tokenizer(input_pretokenized, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                output = model(input_ids=tokenized_input.input_ids.to(device))
            logits = torch.softmax(output.logits, dim=1)  # (bz, 2)
            res.append(logits.squeeze()[-1].item())
        return np.mean(res)
        

def read(ipt, doc_file):
    doc_file = f'/data/disk0/jiawei/na/context-aware-decoding-qfs/data_new_try_this/{doc_file}/{doc_file}_test_len10_notriblock2_order_sample10.jsonl'
    preds, refs = [], []
    with open(ipt, 'r', encoding='utf8') as f:
        for line in f:
            line= json.loads(line.strip())
            preds.append(line["prediction"])
            refs.append(line["reference:"])
    with open(doc_file, 'r', encoding='utf8') as f:
        docs = [json.loads(line.strip())["document"][:1000] for line in f]
        docs = [f"News article: {doc}. Summary of the above news article:" for doc in docs]
    print(preds[0])
    print(docs[0])
    print(len(preds),len(docs))
    return preds, refs, docs
    

if __name__ == "__main__":
    ipt = '/data/disk0/jiawei/na/upload_20241111_na/na/output/gpt-neo-2-7B-512-test.jsonl'
    doc_file = 'cnndm'
    e = Evaluator()
    print(f"{doc_file}: {ipt}")
    
    preds, refs, docs = read(ipt,doc_file)
    e.evaluate(preds, refs, docs)
    