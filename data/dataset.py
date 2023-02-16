from torch.utils.data import Dataset
import numpy as np
import torch
import requests
import newspaper
from bs4 import BeautifulSoup
import textwrap
import re
import os
import tiktoken
from tqdm import tqdm


class PaulGrahamEssaysDataset(Dataset):
    def __init__(self, ctx_size, split='train'):
        data_path = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        
        if os.path.isfile(data_path):
            self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        else:
            self.prepare_dataset()
            self.data = np.memmap(data_path, dtype=np.uint16, mode='r')

        self.ctx_size = ctx_size
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def __len__(self):
        # we don't want to get any index out of range errors
        return len(self.data) - self.ctx_size

    def __getitem__(self, idx):
        # we want a sequence length of 1 more than the ctx_len
        # seq: "hello", "there", "my", "friend"
        # y = ["there", "my", "friend"]
        # x = ["hello", "there", "my"]
        seq = torch.from_numpy((self.data[idx:idx + self.ctx_size + 1]).astype(np.int64))
        return seq[:-1], seq[1:]

    def tokenize(self, example):
            ids = self.tokenizer.encode_ordinary(example)
            ids.append(self.tokenizer.eot_token)

            return {'ids': ids, 'len': len(ids)}

    def prepare_dataset(self):
        print('downloading and processing dataset...')
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

        print('tokenizing essays...')
        tokenized_essays = []
        for essay in tqdm(cleaned_essays):
            tokenized_essays.append(self.tokenize(essay))

        num_train = int(len(tokenized_essays)*0.9)

        train_essays = tokenized_essays[:num_train]
        val_essays = tokenized_essays[num_train:]

        print('writing train/val sets to binary files...')
        # this preprocessing method was inspired by nanoGPT's prepare.py
        splits = {'train': train_essays, 'val': val_essays}
        for split, dataset in tqdm(splits.items()):
            filename = f'{split}.bin'
            dtype = np.uint16

            arr_len = 0
            for essay in dataset:
                arr_len += essay['len']

            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

            idx = 0
            for essay in tqdm(dataset):
                arr[idx : idx + essay['len']] = essay['ids']
                idx += essay['len']
            
            arr.flush()