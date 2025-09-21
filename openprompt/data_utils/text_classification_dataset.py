
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains the logic for loading data for all TextClassification tasks.
"""

import os
import json, csv
import pandas as pd
import csv
from .data_processor import DataProcessor
from .utils import InputExample
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable

from openprompt.utils.logging import logger

from openprompt.data_utils.utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor


class MnliProcessor(DataProcessor):
    # TODO Test needed
    def __init__(self):
        super().__init__()
        self.labels = ["contradiction", "entailment", "neutral"]

    def get_examples(self, data_dir, split):
        # 构建指向 train.csv 或 test.csv 的正确路径
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        # 使用 utf-8 编码打开文件，更加稳妥
        with open(path, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                # 1. 按照您数据的四列格式进行解包
                label, headline, body, pic_add = row
                # 2. 将标题赋值给 text_a
                text_a = headline.replace('\\', ' ')

                # 3. (核心修改) 将正文赋值给 text_b
                text_b = body.replace('\\', ' ')

                # 4. 保留 pic_add 作为元数据
                meta = pic_add

                # 5. 创建包含标题(text_a)和正文(text_b)的 InputExample
                #    同时，标签直接转换为整数，不再减1
                example = InputExample(guid=str(idx), text_a=text_a, text_b=text_b, meta=meta, label=int(label))

                examples.append(example)
        return examples



class AgnewsProcessor(DataProcessor):
    """
    `AG News <https://arxiv.org/pdf/1509.01626.pdf>`_ is a News Topic classification dataset

    we use dataset provided by `LOTClass <https://github.com/yumeng5/LOTClass>`_

    Examples:

    ..  code-block:: python

        from openprompt.data_utils.text_classification_dataset import PROCESSORS

        base_path = "datasets/TextClassification"

        dataset_name = "agnews"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        trainvalid_dataset = processor.get_train_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert processor.get_num_labels() == 4
        assert processor.get_labels() == ["World", "Sports", "Business", "Tech"]
        assert len(trainvalid_dataset) == 120000
        assert len(test_dataset) == 7600
        assert test_dataset[0].text_a == "Fears for T N pension after talks"
        assert test_dataset[0].text_b == "Unions representing workers at Turner   Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul."
        assert test_dataset[0].label == 2
    """

    def __init__(self):
        super().__init__()
        self.labels = ["World", "Sports", "Business", "Tech"]

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            # (for 循环开始)
            for idx, row in enumerate(reader):
                label, headline, body, pic_add = row

                # --- 核心修改：增加对空值的检查和处理 ---
                # 如果 headline 是空的 (Pandas 读取后可能为 NaN), 则将其转换为空字符串
                headline_str = str(headline) if pd.notna(headline) else ""
                # 对 body 做同样的处理
                body_str = str(body) if pd.notna(body) else ""
                # --- 修改结束 ---

                text_a = headline_str.replace('\\', ' ')
                text_b = body_str.replace('\\', ' ')
                meta = pic_add

                # 创建 InputExample, 确保传入的是处理过的字符串
                example = InputExample(guid=str(idx), text_a=text_a, text_b=text_b, meta=meta, label=int(label))
                examples.append(example)
            # (for 循环结束)
            # # ---探针 #1：检查处理器输出 ---
            # if examples:
            #     print("\n" + "=" * 20 + " [探针 #1: 数据处理器] " + "=" * 20)
            #     print(f"CnClickbaitProcessor 已为 '{split}' 数据集生成 {len(examples)} 个样本。")
            #     first_ex = examples[0]
            #     print(f"检查第一个样本 (guid={first_ex.guid}):")
            #     print(f"  - text_a (标题) 前50个字符: {first_ex.text_a[:50]}")
            #     print(f"  - text_b (正文) 是否为空: {'是' if not first_ex.text_b else '否'}")
            #     print(f"  - text_b (正文) 前50个字符: {first_ex.text_b[:50]}")
            #     print("=" * 60 + "\n")
            # # --- 探针结束 ---
        return examples


class CnClickbaitProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        # 这里的标签列表需要与您的 verbalizer 文件中的键完全对应
        self.labels = ["非标题党", "好奇心差距型标题党", "内容失真型标题党"]

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                if len(row) < 4:
                    print(f"警告: 第 {idx + 1} 行数据不完整（少于4列），已跳过。内容: {row}")
                    continue

                label, headline, body, pic_add = row

                if not label or not label.strip():
                    print(f"警告: 第 {idx + 1} 行的标签(label)为空，已跳过该行。")
                    continue

                headline_str = str(headline) if pd.notna(headline) else ""
                body_str = str(body) if pd.notna(body) else ""

                # --- 核心修正 ---
                text_a = headline_str.replace('\\', ' ')
                text_b = body_str.replace('\\', ' ')  # <-- 确保这一行是有效的
                meta = pic_add

                # <-- 确保创建对象时传入了 text_b
                example = InputExample(guid=str(idx), text_a=text_a, text_b=text_b, meta=meta, label=int(label))
                examples.append(example)

        return examples
class DBpediaProcessor(DataProcessor):
    """
    `Dbpedia <https://aclanthology.org/L16-1532.pdf>`_ is a Wikipedia Topic Classification dataset.

    we use dataset provided by `LOTClass <https://github.com/yumeng5/LOTClass>`_

    Examples:

    ..  code-block:: python

        from openprompt.data_utils.text_classification_dataset import PROCESSORS

        base_path = "datasets/TextClassification"

        dataset_name = "dbpedia"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        trainvalid_dataset = processor.get_train_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert processor.get_num_labels() == 14
        assert len(trainvalid_dataset) == 560000
        assert len(test_dataset) == 70000
    """

    def __init__(self):
        super().__init__()
        self.labels = ["company", "school", "artist", "athlete", "politics", "transportation", "building", "river", "village", "animal", "plant", "album", "film", "book",]

    def get_examples(self, data_dir, split):
        examples = []
        label_file  = open(os.path.join(data_dir,"{}_labels.txt".format(split)),'r')
        labels  = [int(x.strip()) for x in label_file.readlines()]
        with open(os.path.join(data_dir,'{}.txt'.format(split)),'r') as fin:
            for idx, line in enumerate(fin):
                splited = line.strip().split(". ")
                text_a, text_b = splited[0], splited[1:]
                text_a = text_a+"."
                text_b = ". ".join(text_b)
                example = InputExample(guid=str(idx), text_a=text_a, text_b=text_b, label=int(labels[idx]))
                examples.append(example)
        return examples


class ImdbProcessor(DataProcessor):
    """
    `IMDB <https://ai.stanford.edu/~ang/papers/acl11-WordVectorsSentimentAnalysis.pdf>`_ is a Movie Review Sentiment Classification dataset.

    we use dataset provided by `LOTClass <https://github.com/yumeng5/LOTClass>`_

    Examples:

    ..  code-block:: python

        from openprompt.data_utils.text_classification_dataset import PROCESSORS

        base_path = "datasets/TextClassification"

        dataset_name = "imdb"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        trainvalid_dataset = processor.get_train_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert processor.get_num_labels() == 2
        assert len(trainvalid_dataset) == 25000
        assert len(test_dataset) == 25000
    """

    def __init__(self):
        super().__init__()
        self.labels = ["negative", "positive"]

    def get_examples(self, data_dir, split):
        examples = []
        label_file = open(os.path.join(data_dir, "{}_labels.txt".format(split)), 'r')
        labels = [int(x.strip()) for x in label_file.readlines()]
        with open(os.path.join(data_dir, '{}.txt'.format(split)),'r') as fin:
            for idx, line in enumerate(fin):
                text_a = line.strip()
                example = InputExample(guid=str(idx), text_a=text_a, label=int(labels[idx]))
                examples.append(example)
        return examples


    @staticmethod
    def get_test_labels_only(data_dir, dirname):
        label_file  = open(os.path.join(data_dir,dirname,"{}_labels.txt".format('test')),'r')
        labels  = [int(x.strip()) for x in label_file.readlines()]
        return labels


# class AmazonProcessor(DataProcessor):
#     """
#     `Amazon <https://cs.stanford.edu/people/jure/pubs/reviews-recsys13.pdf>`_ is a Product Review Sentiment Classification dataset.

#     we use dataset provided by `LOTClass <https://github.com/yumeng5/LOTClass>`_

#     Examples: # TODO implement this
#     """

#     def __init__(self):
#         # raise NotImplementedError
#         super().__init__()
#         self.labels = ["bad", "good"]

#     def get_examples(self, data_dir, split):
#         examples = []
#         label_file = open(os.path.join(data_dir, "{}_labels.txt".format(split)), 'r')
#         labels = [int(x.strip()) for x in label_file.readlines()]
#         if split == "test":
#             logger.info("Sample a mid-size test set for effeciecy, use sampled_test_idx.txt")
#             with open(os.path.join(self.args.data_dir,self.dirname,"sampled_test_idx.txt"),'r') as sampleidxfile:
#                 sampled_idx = sampleidxfile.readline()
#                 sampled_idx = sampled_idx.split()
#                 sampled_idx = set([int(x) for x in sampled_idx])

#         with open(os.path.join(data_dir,'{}.txt'.format(split)),'r') as fin:
#             for idx, line in enumerate(fin):
#                 if split=='test':
#                     if idx not in sampled_idx:
#                         continue
#                 text_a = line.strip()
#                 example = InputExample(guid=str(idx), text_a=text_a, label=int(labels[idx]))
#                 examples.append(example)
#         return examples


class AmazonProcessor(DataProcessor):
    """
    `Amazon <https://cs.stanford.edu/people/jure/pubs/reviews-recsys13.pdf>`_ is a Product Review Sentiment Classification dataset.

    we use dataset provided by `LOTClass <https://github.com/yumeng5/LOTClass>`_

    Examples: # TODO implement this
    """

    def __init__(self):
        super().__init__()
        self.labels = ["bad", "good"]

    def get_examples(self, data_dir, split):
        examples = []
        label_file = open(os.path.join(data_dir, "{}_labels.txt".format(split)), 'r')
        labels = [int(x.strip()) for x in label_file.readlines()]
        with open(os.path.join(data_dir,'{}.txt'.format(split)),'r') as fin:
            for idx, line in enumerate(fin):
                text_a = line.strip()
                example = InputExample(guid=str(idx), text_a=text_a, label=int(labels[idx]))
                examples.append(example)
        return examples


class YahooProcessor(DataProcessor):
    """
    Yahoo! Answers Topic Classification Dataset

    Examples:

    ..  code-block:: python

        from openprompt.data_utils.text_classification_dataset import PROCESSORS

        base_path = "datasets/TextClassification"
    """

    def __init__(self):
        super().__init__()
        self.labels = ["Society & Culture", "Science & Mathematics", "Health", "Education & Reference", "Computers & Internet", "Sports", "Business & Finance", "Entertainment & Music"
                        ,"Family & Relationships", "Politics & Government"]

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, question_title, question_body, answer = row
                text_a = ' '.join([question_title.replace('\\n', ' ').replace('\\', ' '),
                                   question_body.replace('\\n', ' ').replace('\\', ' ')])
                text_b = answer.replace('\\n', ' ').replace('\\', ' ')
                example = InputExample(guid=str(idx), text_a=text_a, text_b=text_b, label=int(label)-1)
                examples.append(example)
        return examples

# class SST2Processor(DataProcessor):
#     """
#     #TODO test needed
#     """

#     def __init__(self):
#         raise NotImplementedError
#         super().__init__()
#         self.labels = ["negative", "positive"]

#     def get_examples(self, data_dir, split):
#         examples = []
#         path = os.path.join(data_dir,"{}.tsv".format(split))
#         with open(path, 'r') as f:
#             reader = csv.DictReader(f, delimiter='\t')
#             for idx, example_json in enumerate(reader):
#                 text_a = example_json['sentence'].strip()
#                 example = InputExample(guid=str(idx), text_a=text_a, label=int(example_json['label']))
#                 examples.append(example)
#         return examples
class SST2Processor(DataProcessor):
    """
    `SST-2 <https://nlp.stanford.edu/sentiment/index.html>`_ dataset is a dataset for sentiment analysis. It is a modified version containing only binary labels (negative or somewhat negative vs somewhat positive or positive with neutral sentences discarded) on top of the original 5-labeled dataset released first in `Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank <https://aclanthology.org/D13-1170.pdf>`_

    We use the data released in `Making Pre-trained Language Models Better Few-shot Learners (Gao et al. 2020) <https://arxiv.org/pdf/2012.15723.pdf>`_

    Examples:

    ..  code-block:: python

        from openprompt.data_utils.lmbff_dataset import PROCESSORS

        base_path = "datasets/TextClassification"

        dataset_name = "SST-2"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        train_dataset = processor.get_train_examples(dataset_path)
        dev_dataset = processor.get_dev_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert processor.get_num_labels() == 2
        assert processor.get_labels() == ['0','1']
        assert len(train_dataset) == 6920
        assert len(dev_dataset) == 872
        assert len(test_dataset) == 1821
        assert train_dataset[0].text_a == 'a stirring , funny and finally transporting re-imagining of beauty and the beast and 1930s horror films'
        assert train_dataset[0].label == 1

    """
    def __init__(self):
        super().__init__()
        self.labels = ['0', '1']

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.tsv")
        examples = []
        with open(path, encoding='utf-8')as f:
            lines = f.readlines()
            for idx, line in enumerate(lines[1:]):
                linelist = line.strip().split('\t')
                text_a = linelist[0]
                label = linelist[1]
                guid = "%s-%s" % (split, idx)
                example = InputExample(guid=guid, text_a=text_a, label=self.get_label_id(label))
                examples.append(example)
        return examples

PROCESSORS = {
    "agnews": AgnewsProcessor,
    "dbpedia": DBpediaProcessor,
    "amazon" : AmazonProcessor,
    "imdb": ImdbProcessor,
    "sst-2": SST2Processor,
    "mnli": MnliProcessor,
    "yahoo": YahooProcessor,
}
