import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartForConditionalGeneration, BartTokenizer


class Text2SQL(Dataset):
    def __init__(self, data: Dataset):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item['text'], item['sql']


with open('c:\\Users\\William\\Desktop\\text2sql.json') as f:
    dataset = Text2SQL(json.load(f, strict=False))

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

for parameter in model.parameters():
    parameter.requires_grad = False

optimizer = torch.optim.Adam(model.classification_head.parameters(), lr=5e-5)
# loss = torch.nn.CrossEntropyLoss()

for epoch in range(3):
    for src, tgt in dataloader:
        print(epoch)
        print(src)
        print(tgt)
        src = tokenizer(src, return_tensors='pt', padding=True)
        tgt = tokenizer(tgt, return_tensors='pt', padding=True)
        output = model(input_ids=src['input_ids'], labels=tgt['input_ids'])
        loss = output.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
