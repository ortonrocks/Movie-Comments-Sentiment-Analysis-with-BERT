
#!pip install pytorch_pretrained_bert
import os

import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertTokenizer

# 识别的类型
key = {   0: '0',
       1: '1',
       2: '2',
       3: '3',
       4: '4',
       5: '5'
       }


class Config:
    """配置参数"""

    def __init__(self):
        cru = './'
        self.class_list = [str(i) for i in range(len(key))]  # 類別名單
        self.save_path = './THUCNews/saved_dict/bert.ckpt'
        self.device = torch.device('cpu')
        self.require_improvement = 1000  # 如果超過1000 epoch 效果無提升，提前結束訓練
        self.num_classes = len(self.class_list)  # 類別數
        self.num_epochs = 3  # epoch
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句話的長句（長切短補）
        self.learning_rate = 5e-5  # 學習率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768

    def build_dataset(self, text):
        lin = text.strip()
        pad_size = len(lin)
        token = self.tokenizer.tokenize(lin)
        token = ['[CLS]'] + token
        token_ids = self.tokenizer.convert_tokens_to_ids(token)
        mask = [1] * pad_size
        token_ids = token_ids[:pad_size]
        return torch.tensor([token_ids], dtype=torch.long), torch.tensor([mask])


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]
        mask = x[1]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out


config = Config()
model = Model(config).to(config.device)
model.load_state_dict(torch.load(config.save_path, map_location='cpu'))


def prediction_model(text):

    data = config.build_dataset(text)
    with torch.no_grad():
        outputs = model(data)
        num = torch.argmax(outputs)
    return key[int(num)]


if __name__ == '__main__':
#輸入一句話預測
    print(prediction_model('要知道劇中裡的這些要素真的很不容易，也謝謝漫威和索尼讓我從小看到大'))