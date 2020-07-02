import os
import csv
import numpy as np
import requests
import chardet
from fake_useragent import UserAgent
from bs4 import BeautifulSoup

query_dataset_csv_path = os.path.join(os.path.abspath('.'), 'model_train_data', 'tbrain_train_final_0610.csv')

def set_header_user_agent():
  user_agent = UserAgent(use_cache_server=False)
  return user_agent.random

#** 讀資料集來源csv檔 **#
with open(query_dataset_csv_path, newline='', encoding='utf-8') as csvfile:
  
  print('讀資料集來源csv檔')

  # 讀取 CSV 檔內容，將每一列轉成一個 dictionary
  rows = csv.DictReader(csvfile)

  user_agent = set_header_user_agent()

  #** 爬蟲開始 **#
  # 以迴圈輸出指定欄位
  for row in rows:

    # print(row['news_ID'], row['hyperlink'], row['content'], row['name'])
    try:
      req = requests.get(row['hyperlink'], timeout=10, headers={ 'user-agent': user_agent })
      req_content = req.content
      detector = chardet.detect(req_content)
      print(row['news_ID'], ' ', detector['encoding'].lower())
      print("監測到的detector資料：{0}".format(detector))
      req_text = req_content.decode(detector['encoding'])
    except requests.exceptions.Timeout:
      print('Read Timeout')
    except UnicodeDecodeError:
      if detector['encoding'].lower() == 'big5':
        req_text = req_content.decode('big5-hkscs')
      else:
        try: 
          req_text = req_content.decode('utf-8')
        except:
          print('error page: ', row['news_ID'], row['hyperlink'])
          continue

    soup = BeautifulSoup(req_text, 'html.parser')

    find_content = ''
    find_p_elements = soup.findAll('p')
    if find_p_elements is None:
      continue
    for find_p_element in find_p_elements:
      find_text = find_p_element.text
      find_content += find_text

    find_pre_elements = soup.findAll('pre')
    if find_pre_elements is None:
      continue
    for find_pre_element in find_pre_elements:
      find_text = find_pre_element.text
      find_content += find_text
    
    find_content = find_content.replace('\r\n', '<newline>').replace('\n', '<newline>')

    print('Finish crawling ', row['news_ID'], row['hyperlink'])

    #** 存爬蟲結果csv檔 **#
    crawler_data_folder = 'crawler_data'
    if not os.path.exists(crawler_data_folder):
        os.mkdir(crawler_data_folder)

    with open(os.path.join(crawler_data_folder, 'crawler_data.csv'), mode='a+', newline='', encoding='utf-8') as csvfile:
      fieldnames = ['news_ID', 'hyperlink', 'content', 'name']
      news_writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',', quotechar='"')
      news_writer.writerow({'news_ID': row['news_ID'], 'hyperlink': row['hyperlink'], 'content': find_content, 'name': row['name']})










