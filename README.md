# Movie-Comments-Sentiment-Analysis-with-BERT

BERT中文電影評論分類

參考：https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch.git

利用BERT建立中文電影分類模型

使用前請先下載以下資料：

1.pytorch_model.bin:https://drive.google.com/file/d/19z7FZI7nbQ79nPg5myITxPtvJLkXSnS3/view?usp=sharing
並放置bert_pretrain資料夾中

2.BERT model: https://drive.google.com/file/d/1Oe13TGsbbiesaYZigVsdIjab3Zj1WiUR/view?usp=sharing
並放置THUCNews\saved_dict資料夾中

在BERT_Predict.py中最下面輸入預測電影評論即可獲得預測分數：（最低1，最高5）

如圖：

![alt text](https://github.com/ortonrocks/Movie-Comments-Sentiment-Analysis-with-BERT/blob/main/BERT.jpg.png)

資料訓練集：

![alt text](https://github.com/ortonrocks/Movie-Comments-Sentiment-Analysis-with-BERT/blob/main/training_datasets.png)

BERT訓練結果：

![alt text](https://github.com/ortonrocks/Movie-Comments-Sentiment-Analysis-with-BERT/blob/main/training_results.png)



