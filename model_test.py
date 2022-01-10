import os
import re
import math
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoModel, BertTokenizerFast


device = torch.device("cpu")

print("Finished importing dependencies")

#Function to determine overall sentiment based on existing labels
def determineOverall(arr) -> int:
  res = 0
  positive = {5}
  negative = {1, 3, 4, 6}
  neutral = {0, 2, 7, 8}
  for num in arr:
    if num in positive:
      res += 1
    elif num in negative:
      res -= 1
    elif num in neutral:
      res += 0
  if res > 0:
    return 2
  elif res < 0:
    return 1
  elif res  == 0:
    return 0  

#converting the array of strings to ints
def convArr(arr):
  # print(arr)
  stoi = lambda x: int(x)
  for i in range(len(arr)):
    arr[i] = stoi(arr[i])

col_list = ["Subtitle", "Labels","Language"]  #three columns seperating the dialouges, language and its emotions (emotions numbered from 1 to 8 according to  Plutchik emotions )
df = pd.read_csv("./combined_tsv.tsv",sep="\t", usecols=col_list)


newLst = [] # build newLst with above function
count = 0
df = df.drop(index=164453)  #weird text causing convArr to fail
for ind in df.index:
  a = df["Labels"][ind]
  b = a.split(", ")
  convArr(b)
  res = determineOverall(b)
  newLst.append(res)
  
df["Overall"] = newLst
df = df.dropna()

#splitting into train, validation and test sets
train_text, temp_text, train_labels, temp_labels = train_test_split(df["Subtitle"], df["Overall"], random_state=2018, test_size=0.2, stratify=df["Overall"])
# df.groupby(['year', 'month', 'class']).size()
val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, random_state=2018, test_size=0.5, stratify=temp_labels)

bert = AutoModel.from_pretrained('bert-base-multilingual-cased')

tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

max_seq_len = 64

tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = max_seq_len,
    padding = 'max_length',
    truncation=True,
    return_token_type_ids=False
)

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())

for param in bert.parameters():
    param.requires_grad = False

class BERT_Arch(nn.Module):

    def __init__(self, bert):
      
      super(BERT_Arch, self).__init__()

      self.bert = bert 
      
      # dropout layer
      self.dropout = nn.Dropout(0.1)
      
      # relu activation function
      self.relu =  nn.ReLU()

      # dense layer 1
      self.fc1 = nn.Linear(768,64)
      
      # dense layer 2 (Output layer)
      self.fc2 = nn.Linear(64,3)

      #softmax activation function
      self.softmax = nn.LogSoftmax()

    #define the forward pass
    def forward(self, sent_id, mask):

      #pass the inputs to the model  
      _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
      
      x = self.fc1(cls_hs)

      x = self.relu(x)

      x = self.dropout(x)

      # output layer
      x = self.fc2(x)
      
      # apply softmax activation
      x = self.softmax(x)

      return x

model = BERT_Arch(bert)

# # pushing the model to GPU
model = model.to(device)

path = './saved_weights.pt'
model.load_state_dict(torch.load(path, map_location=device))

with torch.no_grad():
  preds = model(test_seq.to(device), test_mask.to(device))
  preds = preds.detach().cpu().numpy()
