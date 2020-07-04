import os
import logging
import re
import pandas as pd
import csv
import jieba
from stanfordcorenlp import StanfordCoreNLP
from opencc import OpenCC
from gensim.models import word2vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# global variable
crawler_data_csv_path = os.path.join(
    os.path.abspath('..'), 'crawler_data', 'crawler_data.csv')
stanfordcorenlp_model_path = os.path.join(
    os.path.abspath('.'), 'lib', 'stanford-corenlp-4.0.0')

df = pd.read_csv(crawler_data_csv_path)
print(df.head())

with open(crawler_data_csv_path, newline='', encoding='utf-8') as csvfile:

    print('讀資料集來源csv檔')

    # 讀取 CSV 檔內容，將每一列轉成一個 dictionary
    rows = csv.DictReader(csvfile)
    index = 0
    for row in rows:
        if index == 5:
            break
        index += 1
        print('Tokenize第', row['news_ID'], '則新聞')
        
        # 使用jieba斷詞

        # jieba_cut = list(jieba.cut(row['content'], cut_all=False))
        # print(" / ".join(jieba_cut))
        # train_data = [word for word in jieba_cut if word != '']
        # train_data = ' '.join(train_data)
        # open('jieba_seg.txt', 'w', encoding='utf-8').write(train_data)

        # Preset
        nlp = StanfordCoreNLP(stanfordcorenlp_model_path, lang='zh', memory='8g')
        cc = OpenCC('tw2sp')
        article = cc.convert(row['content'])

        # 命名實體識別
        print('命名實體識別：', nlp.ner(article))

        # 使用StanfordCoreNLP斷詞
        nlp_cut = nlp.word_tokenize(article)
        nlp.close()
        print(" / ".join(nlp_cut))

        # 儲存斷詞結果
        train_data = [word for word in nlp_cut if word != '']
        train_data = ' '.join(train_data)
        open('nlp_seg.txt', 'a', encoding='utf-8').write(train_data)

        
    # model = gensim.models.Word2Vec(sentences=jieba_cut, min_count=10, size=200, workers=4)
    # Settings
    seed = 666
    sg = 0
    window_size = 10
    vector_size = 100
    min_count = 1
    workers = 8
    epochs = 5
    batch_words = 10000

    # 讀取斷詞結果
    train_data = word2vec.LineSentence('nlp_seg.txt')

    model = word2vec.Word2Vec(
        train_data,
        min_count=min_count,
        size=vector_size,
        workers=workers,
        iter=epochs,
        window=window_size,
        sg=sg,
        seed=seed,
        batch_words=batch_words
    )
    # 儲存模型
    model.save('word2vec.model')
    # 載入模型
    model = word2vec.Word2Vec.load('word2vec.model')
    print(model['报酬率'].shape)

    for item in model.most_similar('报酬率'):
        print(item)
    
