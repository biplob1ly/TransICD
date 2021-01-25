import logging
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.corpus import stopwords
from utils import *
from constants import *


def remove_stopwords(text):
    stpwords = set([stopword for stopword in stopwords.words('english')])
    stpwords.update({'admission', 'birth', 'date', 'discharge', 'service', 'sex'})
    tokens = text.strip().split()
    tokens = [token for token in tokens if token not in stpwords]
    return ' '.join(tokens)


def load_dataset(data_setting, batch_size, split):
    data = pd.read_csv(f'{GENERATED_DIR}/{split}_{data_setting}.csv', dtype={'LENGTH': int})
    len_stat = data['LENGTH'].describe()
    logging.info(f'{split} set length stats:\n{len_stat}')

    if data_setting == FULL:
        code_df = pd.read_csv(f'{CODE_FREQ_PATH}', dtype={'code': str})
        all_codes = ';'.join(map(str, code_df['code'].values.tolist()))
        data = data.append({'HADM_ID': -1, 'TEXT': 'remove', 'LABELS': all_codes, 'length': 6},
                        ignore_index=True)

    mlb = MultiLabelBinarizer()
    data['LABELS'] = data['LABELS'].apply(lambda x: str(x).split(';'))
    code_counts = list(data['LABELS'].str.len())
    avg_code_counts = sum(code_counts)/len(code_counts)
    logging.info(f'In {split} set, average code counts per discharge summary: {avg_code_counts}')
    mlb.fit(data['LABELS'])
    temp = mlb.transform(data['LABELS'])
    if mlb.classes_[-1] == 'nan':
        mlb.classes_ = mlb.classes_[:-1]
    logging.info(f'Final number of labels/codes: {len(mlb.classes_)}')

    for i, x in enumerate(mlb.classes_):
        data[x] = temp[:, i]
    data.drop(['LABELS', 'LENGTH'], axis=1, inplace=True)

    if data_setting == FULL:
        data = data[:-1]

    code_list = list(mlb.classes_)
    label_freq = list(data[code_list].sum(axis=0))
    hadm_ids = data['HADM_ID'].values.tolist()
    texts = data['TEXT'].values.tolist()
    labels = data[code_list].values.tolist()
    item_count = (len(texts) // batch_size) * batch_size
    logging.info(f'{split} set true item count: {item_count}\n\n')
    return {'hadm_ids': hadm_ids[:item_count],
            'texts': texts[:item_count],
            'targets': labels[:item_count],
            'labels': code_list,
            'label_freq': label_freq}


def get_all_codes(train_path, dev_path, test_path):
    all_codes = set()
    splits_path = {'train': train_path, 'dev': dev_path, 'test': test_path}
    for split, file_path in splits_path.items():
        split_df = pd.read_csv(file_path, dtype={'HADM_ID': str})
        split_codes = set()
        for codes in split_df['LABELS'].values:
            for code in str(codes).split(';'):
                split_codes.add(code)

        logging.info(f'{split} set has {len(split_codes)} unique codes')
        all_codes.update(split_codes)

    logging.info(f'In total, there are {len(all_codes)} unique codes')
    return list(all_codes)


def load_datasets(data_setting, batch_size):
    train_raw = load_dataset(data_setting, batch_size, split='train')
    dev_raw = load_dataset(data_setting, batch_size, split='dev')
    test_raw = load_dataset(data_setting, batch_size, split='test')

    if train_raw['labels'] != dev_raw['labels'] or dev_raw['labels'] != test_raw['labels']:
        raise ValueError(f"Train dev test labels don't match!")

    return train_raw, dev_raw, test_raw


def load_embedding_weights():
    W = []
    # PAD and UNK already in embed file

    with open(EMBED_FILE_PATH) as ef:
        for line in ef:
            line = line.rstrip().split()
            vec = np.array(line[1:]).astype(np.float)
            # vec = vec / float(np.linalg.norm(vec) + 1e-6)
            W.append(vec)
    logging.info(f'Total token count (including PAD, UNK) of full preprocessed discharge summaries: {len(W)}')
    weights = torch.tensor(W, dtype=torch.float)
    return weights


def load_label_embedding(labels, pad_index):
    code_desc = []
    desc_dt = {}
    max_desc_len = 0
    with open(f'{CODE_DESC_VECTOR_PATH}', 'r') as fin:
        for line in fin:
            items = line.strip().split()
            code = items[0]
            if code in labels:
                desc_dt[code] = list(map(int, items[1:]))
                max_desc_len = max(max_desc_len, len(desc_dt[code]))
    for code in labels:
        pad_len = max_desc_len - len(desc_dt[code])
        code_desc.append(desc_dt[code] + [pad_index] * pad_len)

    code_desc = torch.tensor(code_desc, dtype=torch.long)
    return code_desc


def index_text(data, indexer, max_len, split):
    data_indexed = []
    lens = []
    oov_word_frac = []
    for text in data:
        num_oov_words = 0
        text_indexed = [indexer.index_of(PAD_SYMBOL)]*max_len
        tokens = text.split()
        text_len = max_len if len(tokens) > max_len else len(tokens)
        lens.append(text_len)
        for i in range(text_len):
            if indexer.index_of(tokens[i]) >= 0:
                text_indexed[i] = indexer.index_of(tokens[i])
            else:
                num_oov_words += 1
                text_indexed[i] = indexer.index_of(UNK_SYMBOL)
        oov_word_frac.append(num_oov_words/text_len)
        data_indexed.append(text_indexed)
    logging.info(f'{split} dataset has on average {sum(oov_word_frac)/len(oov_word_frac)} oov words per discharge summary')
    return data_indexed, lens


class ICD_Dataset(Dataset):
    def __init__(self, hadm_ids, texts, lens, labels):
        self.hadm_ids = hadm_ids
        self.texts = texts
        self.lens = lens
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def get_code_count(self):
        return len(self.labels[0])

    def __getitem__(self, index):
        hadm_id = torch.tensor(self.hadm_ids[index])
        text = torch.tensor(self.texts[index], dtype=torch.long)
        length = torch.tensor(self.lens[index], dtype=torch.long)
        codes = torch.tensor(self.labels[index], dtype=torch.float)
        return {'hadm_id': hadm_id, 'text': text, 'length': length, 'codes': codes}


def prepare_datasets(data_setting, batch_size, max_len):
    train_data, dev_data, test_data = load_datasets(data_setting, batch_size)
    input_indexer = Indexer()
    input_indexer.add_and_get_index(PAD_SYMBOL)
    input_indexer.add_and_get_index(UNK_SYMBOL)
    with open(VOCAB_FILE_PATH, 'r') as fin:
        for line in fin:
            word = line.strip()
            input_indexer.add_and_get_index(word)

    logging.info(f'Size of training vocabulary including PAD, UNK: {len(input_indexer)}')

    train_text_indexed, train_lens = index_text(train_data['texts'], input_indexer, max_len, split='train')
    dev_text_indexed, dev_lens = index_text(dev_data['texts'], input_indexer, max_len, split='dev')
    test_text_indexed, test_lens = index_text(test_data['texts'], input_indexer, max_len, split='test')

    train_set = ICD_Dataset(train_data['hadm_ids'], train_text_indexed, train_lens, train_data['targets'])
    dev_set = ICD_Dataset(dev_data['hadm_ids'], dev_text_indexed, dev_lens, dev_data['targets'])
    test_set = ICD_Dataset(test_data['hadm_ids'], test_text_indexed, test_lens, test_data['targets'])
    return train_set, dev_set, test_set, train_data['labels'], train_data['label_freq'], input_indexer
