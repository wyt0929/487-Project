import torch
from torch.utils.data import Dataset
import gensim.downloader
import os
import json
from nltk.tokenize import word_tokenize
from transformers import BertModel, BertTokenizer

class LSTMDataset(Dataset):
    def __init__(self, tsv):
        self.embeddings = []
        self.labels = []
        self.vectorizer = gensim.downloader.load('glove-wiki-gigaword-200')
        for i, row in tsv.iterrows():
            ID = row['ID']
            self.labels.append(row['bias'])
            path = os.path.join("processed_jsons",f"{ID}.json")
            file = open(path, 'r', encoding='utf-8')
            text = json.load(file)['content']
            tokens = word_tokenize(text)
            vector = []
            for token in tokens:
                if token in self.vectorizer.key_to_index:
                    vector.append(torch.tensor(self.vectorizer[token]))
            vector = torch.stack(vector)
            self.embeddings.append(vector)
        self.labels = torch.tensor(self.labels,dtype=torch.long)
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class BERTDataset(Dataset):
    def __init__(self, tsv):
        self.embeddings = []
        self.labels = []
        self.vectorizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for i, row in tsv.iterrows():
            ID = row['ID']
            self.labels.append(row['bias'])
            path = os.path.join("processed_jsons",f"{ID}.json")
            file = open(path, 'r', encoding='utf-8')
            text = json.load(file)['content']
            inputs = self.vectorizer(text, padding=True, truncation=True, return_tensors="pt")
            input_ids = inputs['input_ids']          
            attention_mask = inputs['attention_mask']  
            with torch.no_grad(): 
                self.embeddings.append(torch.squeeze(self.bert(input_ids,attention_mask=attention_mask).last_hidden_state[:, 0, :]))
        self.labels = torch.tensor(self.labels,dtype=torch.long)
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class SVMDataset(Dataset):
    def __init__(self, tsv):
        self.embeddings = []
        self.labels = []
        self.vectorizer = gensim.downloader.load('glove-wiki-gigaword-200')
        for i, row in tsv.iterrows():
            ID = row['ID']
            self.labels.append(row['bias'])
            path = os.path.join("processed_jsons",f"{ID}.json")
            file = open(path, 'r', encoding='utf-8')
            text = json.load(file)['content']
            tokens = word_tokenize(text)
            vector = []
            for token in tokens:
                if token in self.vectorizer.key_to_index:
                    vector.append(torch.tensor(self.vectorizer[token]))
            vector = torch.mean(torch.stack(vector),dim=0)
            self.embeddings.append(vector)
        self.labels = torch.tensor(self.labels,dtype=torch.long)
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]