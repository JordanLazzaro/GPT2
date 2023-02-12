import requests
import newspaper
from bs4 import BeautifulSoup
import textwrap
import re
import tiktoken
import os
import numpy as np
import tqdm


URL = "http://paulgraham.com/articles.html"
page = requests.get(URL)

soup = BeautifulSoup(page.content, "html.parser")

print('collecting essay URLs...')
urls = []
for a in tqdm(soup.find_all('a')):
    if 'html' in a['href'] and 'index' not in a['href']:
        urls.append(f"http://paulgraham.com/{a['href']}")

print('scraping essays...')
essay_texts = []
for url in tqdm(urls):
    article = newspaper.Article(url, fetch_images=False)
    article.download()
    article.parse()
    essay_texts.append(article.text)

print('cleaning essay text...')
cleaned_essays = []
for text in tqdm(essay_texts):
    # each section is separated by one or more newline chars
    sections = re.split("[\n]{2,}", text)
    # get rid of empty sections from multiple newlines and wrap to make more legable
    cleaned_sections = [textwrap.fill(section, 60) for section in sections if len(section) > 0]
    # concatinate sections with extra new line
    cleaned_essay = '\n\n'.join(cleaned_sections)
    # fix spacing with reference numbers
    cleaned_essay = cleaned_essay.replace('[ ', '[').replace(' ]', ']')
    cleaned_essays.append(cleaned_essay)

enc = tiktoken.get_encoding("gpt2")
def process(example, enc):
    ids = enc.encode_ordinary(example)
    ids.append(enc.eot_token)

    return {'ids': ids, 'len': len(ids)}

print('tokenizing essays...')

tokenized_essays = []
for essay in tqdm(cleaned_essays):
    tokenized_essays.append(process(essay, enc))

num_train = int(len(tokenized_essays)*0.9)

train_essays = tokenized_essays[:num_train]
val_essays = tokenized_essays[num_train:]

print('writing train/val sets to binary files...')
splits = {'train': train_essays, 'val': val_essays}
for split, dataset in splits.items():
    filename = f'{split}.bin'
    dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)

    arr_len = 0
    for essay in dataset:
        arr_len += essay['len']

    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

    idx = 0
    for essay in tqdm(dataset):
        arr[idx : idx + essay['len']] = essay['ids']
        idx += essay['len']
    
    arr.flush()