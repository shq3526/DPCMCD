from pickle import FALSE
from torch.utils.data.sampler import RandomSampler
from transformers.configuration_utils import PretrainedConfig
from transformers.generation_utils import GenerationMixin
import torch
#print(torch.cuda.is_available())
import torch.nn as nn
from torch.utils.data import Dataset
from typing import *
from openprompt.data_utils import InputExample, InputFeatures
from torch.utils.data._utils.collate import default_collate
from tqdm.std import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.dummy_pt_objects import PreTrainedModel
from openprompt.plms.utils import TokenizerWrapper
from openprompt.prompt_base import Template, Verbalizer
from collections import defaultdict
from openprompt.utils import round_list, signature
import numpy as np
from torch.utils.data import DataLoader
from yacs.config import CfgNode
from openprompt.utils.logging import logger
from transformers import  AdamW, get_linear_schedule_with_warmup
import json
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import BeitImageProcessor, BeitForImageClassification
import ast

from torch_geometric.nn import GATConv
import torch.nn.functional as F
from torch_geometric.data import Data
from ltp import LTP
from transformers import BertTokenizer
import spacy
import torch
from torch_geometric.data import Data

class PromptDataLoader(object):
    r"""
    PromptDataLoader wraps the original dataset. The input data is firstly wrapped with the
    prompt's template, and then is tokenized by a wrapperd-tokenizer.

    Args:
        dataset (:obj:`Dataset` or :obj:`List`): Either a DatasetObject or a list containing the input examples.
        template (:obj:`Template`): A derived class of :obj:`Template`
        tokenizer (:obj:`PretrainedTokenizer`): The pretrained tokenizer.
        tokenizer_wrapper_class (:cls:`TokenizerWrapper`): The class of tokenizer wrapper.
        max_seq_length (:obj:`int`, optional): The max sequence length of the input ids. It's used to truncate sentences.
        batch_size (:obj:`int`, optional): The batch_size of data loader
        teacher_forcing (:obj:`bool`, optional): Whether to fill the mask with target text. Set to true in training generation model.
        decoder_max_length (:obj:`int`, optional): the decoder maximum length of an encoder-decoder model.
        predict_eos_token (:obj:`bool`, optional): Whether to predict the <eos> token. Suggest to set to true in generation.
        truncate_method (:obj:`bool`, optional): the truncate method to use. select from `head`, `tail`, `balanced`.
        kwargs  :Other kwargs that might be passed into a tokenizer wrapper.
    """
    def __init__(self,
                 dataset: Union[Dataset, List],
                 template: Template,
                 tokenizer_wrapper: Optional[TokenizerWrapper] = None,
                 tokenizer: PreTrainedTokenizer = None,
                 tokenizer_wrapper_class = None,
                 verbalizer: Optional[Verbalizer] = None,
                 max_seq_length: Optional[str] = 512,
                 batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = False,
                 teacher_forcing: Optional[bool] = False,
                 decoder_max_length: Optional[int] = -1,
                 predict_eos_token: Optional[bool] = False,
                 truncate_method: Optional[str] = "tail",
                 drop_last: Optional[bool] = False,
                 **kwargs,
                ):

        assert hasattr(dataset, "__iter__"), f"The dataset must have __iter__ method. dataset is {dataset}"
        assert hasattr(dataset, "__len__"), f"The dataset must have __len__ method. dataset is {dataset}"
        self.raw_dataset = dataset

        self.wrapped_dataset = []
        self.tensor_dataset = []
        self.template = template
        self.verbalizer = verbalizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.teacher_forcing = teacher_forcing

        if tokenizer_wrapper is None:
            if tokenizer_wrapper_class is None:
                raise RuntimeError("Either wrapped_tokenizer or tokenizer_wrapper_class should be specified.")
            if tokenizer is None:
                raise RuntimeError("No tokenizer specified to instantiate tokenizer_wrapper.")

            tokenizer_wrapper_init_keys = signature(tokenizer_wrapper_class.__init__).args
            prepare_kwargs = {
                "max_seq_length" : max_seq_length,
                "truncate_method" : truncate_method,
                "decoder_max_length" : decoder_max_length,
                "predict_eos_token" : predict_eos_token,
                "tokenizer" : tokenizer,
                **kwargs,
            }

            to_pass_kwargs = {key: prepare_kwargs[key] for key in prepare_kwargs if key in tokenizer_wrapper_init_keys}
            self.tokenizer_wrapper = tokenizer_wrapper_class(**to_pass_kwargs)
        else:
            self.tokenizer_wrapper = tokenizer_wrapper

        # check the satisfiability of each component
        assert hasattr(self.template, 'wrap_one_example'), "Your prompt has no function variable \
                                                         named wrap_one_example"

        # process
        self.wrap()
        self.tokenize()

        if self.shuffle:
            sampler = RandomSampler(self.tensor_dataset)
        else:
            sampler = None

        self.dataloader = DataLoader(
            self.tensor_dataset,
            batch_size = self.batch_size,
            sampler= sampler,
            collate_fn = InputFeatures.collate_fct,
            drop_last = drop_last,
        )


    def wrap(self):
        r"""A simple interface to pass the examples to prompt, and wrap the text with template.
        """
        if isinstance(self.raw_dataset, Dataset) or isinstance(self.raw_dataset, List):
            assert len(self.raw_dataset) > 0, 'The dataset to be wrapped is empty.'
            # for idx, example in tqdm(enumerate(self.raw_dataset),desc='Wrapping'):
            for idx, example in enumerate(self.raw_dataset):
                if self.verbalizer is not None and hasattr(self.verbalizer, 'wrap_one_example'): # some verbalizer may also process the example.
                    example = self.verbalizer.wrap_one_example(example)
                # print(example['meta'])
                '''
                [[{'text': '', 'soft_token_ids': 1, 'loss_ids': 0, 'shortenable_ids': 0}, 
                {'text': '<mask>', 'soft_token_ids': 0, 'loss_ids': 1, 'shortenable_ids': 0}, 
                {'text': ' 新闻:', 'soft_token_ids': 0, 'loss_ids': 0, 'shortenable_ids 0}, 
                {'text': ' 死刑！湖南6岁男童遭邻居砍死案一审宣判，法院：未发现有精神病', 'soft_token_ids': 0, 'loss_ids': 0, 'shortenable_ids': 1}, 
                {'text': tenable_ids': 1}], {'guid': '3', 'label': 0, 'meta': "['./zongxiang/not_clickbait/555/1.png', './zongxiang/not_clickbait/555/2.png', './zongxiang/not_clickbait/555/3.png']"}]

                '''
                wrapped_example = self.template.wrap_one_example(example)
                # print(wrapped_example[1]['meta'])
                self.wrapped_dataset.append(wrapped_example)
        else:
            raise NotImplementedError

    def tokenize(self) -> None:
        r"""Pass the wrapped text into a prompt-specialized tokenizer,
           the true PretrainedTokenizer inside the tokenizer is flexible, e.g. AlBert, Bert, T5,...
        """
        for idx, wrapped_example in tqdm(enumerate(self.wrapped_dataset),desc='tokenizing'):
        # for idx, wrapped_example in enumerate(self.wrapped_dataset):
            inputfeatures = InputFeatures(**self.tokenizer_wrapper.tokenize_one_example(wrapped_example, self.teacher_forcing), **wrapped_example[1]).to_tensor()
            pic_add = wrapped_example[1]['meta']
            inputfeatures['meta'] = pic_add
            title = wrapped_example[0][3]['text']

            inputfeatures['title'] = title
            self.tensor_dataset.append(inputfeatures)

    def __len__(self):
        return  len(self.dataloader)

    def __iter__(self,):
        return self.dataloader.__iter__()



class PromptModel(nn.Module):
    r'''``PromptModel`` is the encapsulation of ``Template`` and the ``pre-trained model``,
    with OpenPrompt, these modules could be flexibly combined. And this class is the base class of ``PromptForClassification`` and ``PromptForGeneration``

    Args:
        plm (:obj:`PreTrainedModel`): The pre-trained language model for the current prompt-learning task.
        template (:obj:`Template`): The ``Template`` object to warp the input data.
        freeze_plm (:obj:`bool`): whether or not to freeze the pretrained language model
        plm_eval_mode (:obj:`bool`): this is a stronger freezing mode than freeze_plm, i.e. the dropout of the model is turned off. No matter whether the other part is set to train.
    '''
    def __init__(self,
                 plm: PreTrainedModel,
                 template: Template,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False,
                ):
        super().__init__()
        self.plm = plm
        self.template = template
        self.freeze_plm = freeze_plm
        self.plm_eval_mode = plm_eval_mode
        if freeze_plm:
            for param in self.plm.parameters():
                param.requires_grad = False
        if plm_eval_mode:
            self.plm.eval()
            for param in self.plm.parameters():
                param.requires_grad = False

        # get model's forward function's keywords
        self.forward_keys = signature(self.plm.forward).args

        self._prepare_main_input_name()

    def _prepare_main_input_name(self):
        model = self.plm
        if hasattr(model, "encoder") and hasattr(model.encoder, "main_input_name"):
            if model.encoder.main_input_name != model.main_input_name:
                main_input_name = model.encoder.main_input_name
            else:
                main_input_name = model.main_input_name
        else:
            main_input_name = getattr(model, "main_input_name", "input_ids")
        self.main_input_name = main_input_name

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            if not (self.plm_eval_mode and 'plm' in name and mode):
                module.train(mode)
        return self

    def forward(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
        r"""
        This is a forward method to make wrapped input data go through the model, and return the output logits.
        Typically, this function aims to predict the ``<mask>`` position.

        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of batchified data sequences.
        """
        # ### PROBE (print): 检查 image_features 是否成功传入 forward
        # if 'image_features' in batch and batch['image_features'] is not None:
        #     print(f"### PROBE (print): 模型 forward() 接收到 image_features, 形状为: {batch['image_features'].shape}, 设备为: {batch['image_features'].device}")
        # else:
        #     print(f"### PROBE (print): 模型 forward() 未接收到 image_features 或其值为 None")

        batch = self.template.process_batch(batch)
        input_batch = {key: batch[key] for key in batch if key in self.forward_keys}
        outputs = self.plm(**input_batch, output_hidden_states=True)
        outputs = self.template.post_processing_outputs(outputs)
        return outputs

    def prepare_model_inputs(self, batch: Union[Dict, InputFeatures]) -> Dict:
        r"""Will be used in generation
        """
        batch = self.template.process_batch(batch)
        input_batch = {key: batch[key] for key in batch if key in self.forward_keys}
        return input_batch

class PromptForClassification(nn.Module):
    r'''``PromptModel`` with a classification head on top. The classification head will map
    the logits in all position of the sequence (return value of a ``PromptModel``) into the
    logits of the labels, using a verbalizer.

    Args:
        plm (:obj:`PretrainedModel`): A pre-traiend model you decide to use for classification, e.g. BERT.
        template (:obj:`Template`): A ``Template`` object you use to wrap the input text for classification, e.g. ``ManualTemplate``.
        verbalizer (:obj:`Verbalizer`): A ``Verbalizer`` object you use to project the labels to label words for classification, e.g. ``ManualVerbalizer``.
        freeze_plm (:obj:`bool`): whether or not to freeze the pretrained language model
        plm_eval_mode (:obj:`bool`): this is a stronger freezing mode than freeze_plm, i.e. the dropout of the model is turned off. No matter whether the other part is set to train.
    '''
    def __init__(self,
                 plm: PreTrainedModel,
                 template: Template,
                 verbalizer: Verbalizer,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False,
                 in_channels: int = 64,
                 out_channels: int = 128
                ):
        super().__init__()
        self.gat = GATConv(in_channels=64, out_channels=out_channels, heads=8)

        self.prompt_model = PromptModel(plm, template, freeze_plm, plm_eval_mode)
        self.verbalizer = verbalizer

    @property
    def plm(self):
        return self.prompt_model.plm

    @property
    def template(self):
        return self.prompt_model.template

    @property
    def device(self,):
        r"""Register the device parameter."""
        return self.plm.device

    def extract_at_mask(self,
                        outputs: torch.Tensor,
                        batch: Union[Dict, InputFeatures]):
        # The first element of the model's output is the sequence of hidden states
        # Its shape is [batch_size, sequence_length, hidden_size]
        hidden_states = outputs[0]

        # batch['loss_ids'] has shape [batch_size, sequence_length]
        # where the [MASK] positions are marked with 1, others with 0.
        # We create a boolean mask from it.
        mask = batch['loss_ids'] > 0

        # Use the boolean mask to select the hidden states of the [MASK] tokens
        outputs_at_mask = hidden_states[mask]

        return outputs_at_mask

    def title_to_pos_graph(self, title):
        ltp = LTP(
            pretrained_model_name_or_path='/home/mjy/anaconda3/envs/pytorch/lib/python3.9/site-packages/ltp/ltp_model')
        # print(title)
        # 调用ltp
        output = ltp.pipeline([title], tasks=["cws", "pos", "sdp", "dep"])
        cws = output.cws  # 分词
        pos = output.pos  # 词性标注  #{'nt', 'q', 'n', 'a', 'v'}
        dep = output.dep[0]  # 依存句法结构
        graph = {'nodes': set(pos[0]), 'edges': []}
        for i, word in enumerate(cws[0]):
            head = dep['head'][i] - 1  # 依存句法关系中的头
            if head >= 0:  # 头存在（注意有些标点符号头为-1）
                label = dep['label'][i]  # 依存句法关系中的标签
                graph['edges'].append((pos[0][i], label, pos[0][head]))
        # 构建边
        edge_index = []
        for edge in graph['edges']:
            if edge[0] in graph['nodes'] and edge[2] in graph['nodes']:
                src = list(graph['nodes']).index(edge[0])
                dst = list(graph['nodes']).index(edge[2])
                edge_index.append([src, dst])
        # print(edge_index)  #[[1, 0], [4, 2], [3, 2], [2, 0]]
        if edge_index == []:
            edge_index = [[0, 0]]
        edge_index = torch.tensor(edge_index).t().contiguous()
        # x = []
        # for node in graph['nodes']:
        #     x.append(list(graph['nodes']).index(node))
        x = torch.tensor(BertTokenizer.from_pretrained(
            '/home/mjy/anaconda3/envs/pytorch/lib/python3.9/site-packages/openprompt/plms/chinese-roberta-wwm-ext').convert_tokens_to_ids(
            pos[0]), dtype=torch.float)
        # x = x.to(torch.float32)
        data = Data(x=x.to('cuda'), edge_index=torch.as_tensor(edge_index).to('cuda'))
        # data = Data(x=torch.tensor(x).to('cuda'), edge_index=torch.tensor(edge_index).to('cuda'))
        '''
        data:
Data(x=[19], edge_index=[2, 18])

        '''
        # print('data:')
        # print(data)
        return data

    def title_to_en_pos_graph(self, title):
        # Load SpaCy English model
        nlp = spacy.load("en_core_web_sm")

        # Process English title using SpaCy
        doc = nlp(title)

        # Extract information
        pos = [token.pos_ for token in doc]
        dep = [(token.dep_, token.head.text) for token in doc]

        # Build graph representation
        graph = {'nodes': set(pos), 'edges': []}
        for i, token in enumerate(doc):
            head = token.head.text
            label = dep[i][0]
            graph['edges'].append((pos[i], label, head))

        # Edge indexing
        edge_index = []
        for edge in graph['edges']:
            if edge[0] in graph['nodes'] and edge[2] in graph['nodes']:
                src = list(graph['nodes']).index(edge[0])
                dst = list(graph['nodes']).index(edge[2])
                edge_index.append([src, dst])

        if edge_index == []:
            edge_index = [[0, 0]]
        edge_index = torch.tensor(edge_index).t().contiguous()

        # Convert node features using BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained('/home/mjy/wangye/text_classification/model/bert-case')
        x = torch.tensor(tokenizer.convert_tokens_to_ids([token.text for token in doc]), dtype=torch.float)

        # Create PyTorch Data object
        data = Data(x=x.to('cuda'), edge_index=edge_index.to('cuda'))
        # print(data)
        return data
    def encode_images_and_combine_features(self, image_paths, text_descriptions):
        # 加载CLIP模型和处理器
        model_name = "/home/mjy/anaconda3/envs/pic_prompt/lib/python3.9/site-packages/openprompt/model"  # 预训练的CLIP模型
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name).to(device)

        # 编码多张图像
        image_features_list = []


        for image_path, text_description in zip(image_paths, text_descriptions):

            image_path = '/home/mjy/wangye/pic_prompt/PCN_clickbait/datasets/TextClassification/'+ image_path.replace("./", "")
            # print(image_path)
            image = Image.open(image_path)
            # inputs = processor(text=text_description, images=image, return_tensors="pt", padding=True).to(device)
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                # outputs = model(**inputs)
                image_features = model.get_image_features(**inputs)
                # print(image_features.shape)
                # image_features = outputs.logits_per_image  # 图像特征
                image_features_list.append(image_features)

        # 合并多张图像特征
        combined_image_features = torch.cat(image_features_list, dim=0)  # 在第0维上拼接
        # print(combined_image_features)
        return combined_image_features

    # 根据通道数选择不同的均值和标准差
    def get_normalization_parameters(self, num_channels):
        if num_channels == 1:  # 灰度图像
            mean = [0.5]
            std = [0.5]
        elif num_channels == 3:  # 彩色图像
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            raise ValueError("Unsupported number of channels")

        return mean, std

    # 图像预处理函数，动态选择均值和标准差，并将灰度图像转换为伪彩色图像
    def preprocess_image(self, image, target_size=(224, 224)):
        # 获取图像的宽度和高度
        width, height = image.size

        # 获取图像的通道数（默认为3，彩色图像）
        num_channels = 3

        # 检查图像是否为灰度图像
        if len(image.getbands()) == 1:
            num_channels = 1

            # 创建一个新的 RGB 模式图像
            new_image = Image.new("RGB", image.size)
            new_image.paste(image)

            # 使用新的图像
            image = new_image
        elif image.mode == 'LA':
            # 如果图像是 "LA" 模式，将其转换为伪彩色图像
            image = image.convert("RGB")

        # 获取相应的均值和标准差（使用彩色图像的均值和标准差，适应灰度图像也是可以的）
        mean, std = self.get_normalization_parameters(3)  # 使用彩色图像的均值和标准差

        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        # print(image)
        # 如果图像有4通道（RGBA），将其转换为3通道（RGB）
        if image.mode == 'RGBA':
            r, g, b, a = image.split()
            image = Image.merge("RGB", (r, g, b))
        # 应用预处理操作
        image = transform(image)

        return image
    def resnet_extract_and_merge_features(self, image_paths):
        # 加载预训练的ResNet模型
        resnet_model = models.resnet50(pretrained=False)
        resnet_model.load_state_dict(torch.load('/home/mjy/wangye/pic_prompt/PCN_clickbait/model/resnet50-0676ba61.pth'))

        # 移除最后一层全连接层，保留特征提取器部分
        resnet_feature_extractor = torch.nn.Sequential(*list(resnet_model.children())[:-1])

        # 设置模型为评估模式，不进行梯度计算
        resnet_feature_extractor.eval()

        # 图像预处理
        # transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])


        # 初始化一个空的特征列表
        features_list = []

        for image_path in image_paths:
            image_path = '/home/mjy/wangye/pic_prompt/PCN_clickbait/datasets/TextClassification/' + image_path.replace("./", "")
            # 读取图像并进行预处理
            image = Image.open(image_path)
            image = self.preprocess_image(image)
            image = image.unsqueeze(0)  # 添加批次维度

            # 提取图像特征
            with torch.no_grad():
                features = resnet_feature_extractor(image)
                # print(features)
                features_list.append(features)
        # 合并多个图像的特征
        merged_features = torch.cat(features_list, dim=0).to('cuda')

        return merged_features

    def Beit_extract_image_features(self, image_paths):
        # 加载 BEiT 模型和图像处理器
        processor = BeitImageProcessor.from_pretrained('/home/mjy/wangye/pic_prompt/PCN_clickbait/model/beit')
        model = BeitForImageClassification.from_pretrained('/home/mjy/wangye/pic_prompt/PCN_clickbait/model/beit')

        # 用于存储图像特征的列表
        image_features = []

        for image_path in image_paths:
            image_path = '/home/mjy/wangye/pic_prompt/PCN_clickbait/datasets/TextClassification/' + image_path.replace("./", "")
            # print(image_path)
            # 打开图像文件并创建 PIL 图像对象
            image = Image.open(image_path).convert("RGB")
            # image = Image.open(image_path)
            # image = self.preprocess_image(image)
            # 使用图像处理器对图像进行处理
            inputs = processor(images=image, return_tensors="pt")

            # 使用模型进行推理并获取图像特征
            outputs = model(**inputs)
            # image_feature = outputs.logits.to('cuda')
            image_feature = outputs.logits.mean(dim=0).to('cuda')  # 使用均值池化获取编码表示
            # image_feature = outputs.logits
            # print(image_feature.shape)
            # 添加图像特征到列表中
            image_features.append(image_feature.unsqueeze(0))

        # 拼接图像特征

        concatenated_features = torch.cat(image_features, dim=0)
        # print(concatenated_features.shape)   #torch.Size([3000])
        return concatenated_features

    def forward(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
        r"""
        This is the forward method of the VLPVerbalizer.

        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The input data.

        Returns:
            :obj:`torch.Tensor`: The final logits of the classification task.
        """
        # 获取预训练语言模型的输出
        outputs = self.prompt_model(batch)

        # 从模型的输出中提取在 [MASK] 位置的表示
        outputs_at_mask = self.extract_at_mask(outputs, batch)

        # 尝试从batch中获取图像和GAT输入，如果不存在则设为None
        image_features = batch.get('image_features', None)
        gat_inputs = batch.get('gat_inputs', None)

        # --- 1. 处理句法特征 (GAT) ---
        # 仅当 gat_inputs 存在时才进行处理
        if gat_inputs is not None:
            num_vectors_pos = outputs_at_mask.shape[0]
            compressed_vectors_pos = []
            for i in range(min(num_vectors_pos, len(gat_inputs))):  # 添加安全检查
                compressed_vector_pos = torch.mean(gat_inputs[i], dim=0, keepdim=True)
                compressed_vectors_pos.append(compressed_vector_pos)
            pos_concatenated = torch.cat(compressed_vectors_pos, dim=0)

            # 初始特征为文本+句法
            final_features = torch.cat((outputs_at_mask, pos_concatenated), dim=1)
        else:
            # 如果没有GAT输入，则初始特征只有文本
            final_features = outputs_at_mask

        # --- 2. 条件性地处理图像特征 ---
        # 只有在 image_features 列表不为空时，才执行图像处理
        if image_features:
            num_vectors_img = outputs_at_mask.shape[0]
            compressed_vectors_img = []
            for i in range(min(num_vectors_img, len(image_features))):
                image_feature = image_features[i]
                compressed_vector_img = torch.mean(image_feature, dim=0, keepdim=True)
                compressed_vectors_img.append(compressed_vector_img)

            if compressed_vectors_img:
                concatenated_img = torch.cat(compressed_vectors_img, dim=0)
                img_concatenated = concatenated_img.squeeze(-1).squeeze(-1)

                # 如果有图像特征，则在已有特征基础上再拼接图像特征
                final_features = torch.cat((final_features, img_concatenated), dim=1)

        # --- 3. 将最终的特征传递给 Verbalizer ---
        # 确保 final_features 始终是一个有效的张量
        label_words_logits = self.verbalizer.process_outputs(final_features, batch=batch)

        # --- 4. 确保函数明确返回计算结果 ---
        return label_words_logits

    def predict(self):
        pass

    def forward_without_verbalize(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
        outputs = self.prompt_model(batch)
        outputs = self.verbalizer.gather_outputs(outputs)
        outputs_at_mask = self.extract_at_mask(outputs, batch)
        return outputs_at_mask

    @property
    def tokenizer(self):
        r'''Utility property, to get the tokenizer more easily.
        '''
        return self.verbalizer.tokenizer


    ##  We comment this code since it conflict with [OpenDelta](https://github.com/thunlp/OpenDelta)
    # def state_dict(self, *args, **kwargs):
    #     """ Save the model using template, plm and verbalizer's save methods."""
    #     _state_dict = {}
    #     if not self.prompt_model.freeze_plm:
    #         _state_dict['plm'] = self.plm.state_dict(*args, **kwargs)
    #     _state_dict['template'] = self.template.state_dict(*args, **kwargs)
    #     _state_dict['verbalizer'] = self.verbalizer.state_dict(*args, **kwargs)
    #     return _state_dict

    # def load_state_dict(self, state_dict, *args, **kwargs):
    #     """ Load the model using template, plm and verbalizer's load methods."""
    #     if 'plm' in state_dict and not self.prompt_model.freeze_plm:
    #         self.plm.load_state_dict(state_dict['plm'], *args, **kwargs)
    #     self.template.load_state_dict(state_dict['template'], *args, **kwargs)
    #     self.verbalizer.load_state_dict(state_dict['verbalizer'], *args, **kwargs)

    def parallelize(self, device_map=None):
        r"""Parallelize the model across device
        """
        if hasattr(self.plm, "parallelize"):
            self.plm.parallelize(device_map)
            self.device_map = self.plm.device_map
            self.template.cuda()
            self.verbalizer.cuda()
        else:
            raise NotImplementedError("parallelize method was not implemented for this plm.")

    def deparallelize(self):
        r"""Deparallelize the model across device
        """
        if hasattr(self.plm, "deparallelize"):
            self.plm.deparallelize()
            self.device_map = None
            self.template.cpu()
            self.verbalizer.cpu()
        else:
            raise NotImplementedError("parallelize method was not implemented for this plm.")


class PromptForGeneration(nn.Module, GenerationMixin):
    r'''``PromptModel`` with generation loss calculation and generation utils integrated.


    Args:
        plm (:obj:`PretrainedModel`): A pre-traiend model you decide to use for generation, e.g. GPT.
        template (:obj:`Template`): A ``Template`` object you use to wrap the input text for classification, e.g. ``PrefixTuningTemplate``.
        tokenizer (:obj:`Tokenizer`): A ``Tokenizer`` of the current model.
        gen_config (:obj:`CfgNode`): The generation configs to pass into `GenerationMixin.generate <https://huggingface.co/transformers/_modules/transformers/generation_utils.html#GenerationMixin.generate>`_
        freeze_plm (:obj:`bool`): whether or not to freeze the pretrained language model
        plm_eval_mode (:obj:`bool`): this is a stronger freezing mode than freeze_plm, i.e. the dropout of the model is turned off. No matter whether the other part is set to train.
    '''

    def __init__(self,
                 plm: PreTrainedModel,
                 template: Template,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool = False,
                 gen_config: Optional[CfgNode] = None,
                 tokenizer: Optional[PreTrainedTokenizer] = None,
                ):
        super().__init__()
        self.freeze_plm = freeze_plm
        if tokenizer is None:
            assert template.tokenizer is not None, "Tokenizer can't be set from input args or template"
            self.tokenizer = template.tokenizer
        else:
            self.tokenizer = tokenizer
        self.prompt_model = PromptModel(plm, template, freeze_plm, plm_eval_mode)

        self.loss_fct = nn.CrossEntropyLoss(reduction='none')
        self.config = plm.config
        if gen_config:
            for key in gen_config:
                setattr(self.config, key, gen_config[key])
        self.in_generation_function = False

        self.main_input_name = self.prompt_model.main_input_name # for transformers 4.17.0 and higher.

    @property
    def generation_config(self):
        return self.plm.generation_config
    
    @property
    def plm(self):
        return self.prompt_model.plm

    @property
    def template(self):
        return self.prompt_model.template

    @property
    def device(self):
        return self.plm.device
    
    def can_generate(self):
        if hasattr(self.prompt_model.plm, "can_generate"):
            return self.prompt_model.plm.can_generate
        else:
            raise AttributeError("the corresponding plm does not have `can_generate` attribute")

    def shift_logits_and_labels(self,
                                logits,
                                loss_ids,
                                reference_ids):

        r"""
        Left shift the label, and make label of the positions that are
        not loss position to -100, which is the ignore index in pytorch's
        loss function.

        Args:
            logits (:obj:`torch.Tensor`):
            batch (:obj:`InputFeatures`): The input features of batchified data sequences.

        Returns:
            shift_logits (:obj:`torch.Tensor`):
            shift_input_ids (:obj:`List[int]`):

        """

        shift_logits = logits[..., :-1, :].contiguous()
        shift_loss_ids = loss_ids[..., 1:].contiguous()
        shift_input_ids = reference_ids[..., 1:].contiguous()
        shift_input_ids = torch.where(shift_loss_ids>0, shift_input_ids, -100)
        return shift_logits, shift_input_ids

    def forward(self, *args, **kwargs):
        r"""In generation process, it will use the plm's forward function.
        This is because, in the first step we will directly call the process_batch function to
        generate initial input with the template, after that the all template
        have been processed into the past_key_value,
        then we can use the normal generation function.
        In learning process, the forward is linked to ``_forward`` functions.
        in which the loss will be calculated for all the positions in the same time.
        """
        if self.in_generation_function:
            return self.plm.forward(*args, **kwargs)
        else:
            return self._forward(*args, **kwargs)

    def _forward(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
        r"""
        This is the forward method of the training of generation in prompt-learning framework.

        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of batchified data sequences.

        Returns:
            loss(:obj:torch.Tensor): The loss of the current generation procedure.
        """
        if self.config.is_encoder_decoder:
            reference_ids = batch['decoder_input_ids']
        else:
            reference_ids = batch['input_ids']  # in case in some template, these field is dropped
        outputs = self.prompt_model(batch)
        logits = outputs.logits
        logits, labels = self.shift_logits_and_labels(logits, batch['loss_ids'], reference_ids)
        batch_size, seq_len, vocab_size = logits.shape
        loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss = loss.view(batch_size, -1).sum(dim=-1) # TODO support more objectives
        loss = loss.mean()
        return loss

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        """skip model kwargs validation, as all arguments will be filtered before feeding into plm in PromptModel"""
        return


    def generate(self, batch: Union[Dict, InputFeatures], verbose: Optional[bool]=False, **generation_kwargs):
        r""" This function wraps the generate() methods in parent class ``GenerationMixin``.
        Forward uses the ``PretrainedModel``'s forward method.
        generation_kwargs include all the parameters that are passed in to
        ``transformers.generation_util.GenerationMixin.generate``

        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of batchified data sequences.
            verbose (:obj:`Optional[bool]`): Set to true to verbose the generated sentence.

        Returns:
            output_sequences (:obj:`List[torch.Tensor]`): The raw sequences generated by the generation model.
            generated_sentences (:obj:`List[torch.Tensor]`): The generated sentences that have been post-processed.
        """
        input_generation_kwargs = generation_kwargs
        if self.config.is_encoder_decoder:
            loss_ids_start = batch['loss_ids'].argmax(dim=-1)
            assert loss_ids_start.min() == loss_ids_start.max(), "The generation start from different position in a batch."
            batch['decoder_input_ids'] = batch['decoder_input_ids'][:, :loss_ids_start.min()+1]
            input_length = batch['decoder_input_ids'].size(1)
            batch_size = batch['decoder_input_ids'].size(0)

            self.generate_ith_token = 0
            self.in_generation_function = True
            output_sequences = super().generate(**batch, **input_generation_kwargs, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
            self.in_generation_function = False
            output_sequences = output_sequences.cpu().tolist()
            generated_sentences = self.post_processing(output_sequences=output_sequences, input_lengths=input_length)
        else:
            input_length = batch['input_ids'].size(1)
            batch_size = batch['input_ids'].size(0)

            # Currently huggingface transformers only support single sample generation, or padding to the left (instead of the right).
            # because it will only extract the last position of the output
            # generate one_by_one
            if 'input_ids_len' in batch:
                input_real_lens = batch['input_ids_len']
            else:
                input_real_lens = torch.sum((batch['input_ids'] != self.tokenizer.pad_token_id).to(torch.int), dim=-1)
            output_sequences = []
            for instance_id in range(batch_size):
                # remove the pad token
                instance = {key: batch[key][instance_id:instance_id+1][:,:input_real_lens[instance_id]] for key in batch if isinstance(batch[key], torch.Tensor) and batch[key].shape[:2]==torch.Size([batch_size, input_length])}
                self.generate_ith_token = 0
                self.in_generation_function = True
                output_sequence = super().generate(**instance, **input_generation_kwargs, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
                self.in_generation_function = False
                output_sequences.extend(output_sequence.cpu().tolist()) # TODO: to support generate multiple sentence
            generated_sentences = self.post_processing(output_sequences=output_sequences, input_lengths=input_real_lens.cpu().tolist())
        if verbose:
            logger.info(f"Generated:{generated_sentences}")
        return output_sequences, generated_sentences



    def post_processing(self, output_sequences, input_lengths):
        r"""
            Post-process the sequences generated by the generation model.

            Args:
                output_sequences (:obj:`torch.Tensor`): The raw sequences generated by the generation model.
                input_lengths (:obj:`int` or `list`): The length(s) of the input sequence.

            Returns:
                :obj:`List`: The generated sentences that have been post-processed.
        """
        generated_sentences = []
        if type(input_lengths)==int:
            input_lengths = [input_lengths] * len(output_sequences)
        for sent_id, seq in enumerate(output_sequences):
            seq = seq[input_lengths[sent_id]:]

            if hasattr(self.tokenizer, "eos_token") and self.tokenizer.eos_token is not None:
                text_output = self.tokenizer.decode(seq, clean_up_tokenization_spaces=True, skip_special_tokens=False)
                idx = text_output.find(self.tokenizer.eos_token)
                if idx >= 0:
                    text_output = text_output[:idx]
            else:
                text_output = self.tokenizer.decode(seq, clean_up_tokenization_spaces=True, skip_special_tokens=True)
            text_output = text_output.strip()
            generated_sentences.append(text_output)
        return generated_sentences



    def prepare_inputs_for_generation(self, input_ids: Optional[torch.Tensor] = None,
                                         **model_kwargs):
        r"""This function wraps the ``prepare_inputs_for_generation`` function in the huggingface transformers.

        When the `past` not in model_kwargs, we prepare the input from scratch.
        When `past` is in model_kwargs, we don't need to prepare the template wrapped input,
        instead we use the inner pretrain_models' function to prepare the next step's input.
        `model_kwargs` includes all the argument passed in the `batch`: InputFeatures, except ``input_ids``
        , as long as they do not conflict with keywords in ``generation_kwargs``.    if 'past' not in model_kwargs: # the past_key_value not in model_kwargs, then we need to prepare input from scrath
        , as long as they do not conflict with keywords in ``generation_kwargs``.

        Args:
            input_ids(:obj:`torch.Tensor`): Indices of input sequence tokens in the vocabulary.
        """
        if self.generate_ith_token == 0 and 'encoder_outputs' not in model_kwargs: # generating the first token in decoder only setting.

            batch = InputFeatures(input_ids=input_ids, **model_kwargs)
            model_inputs = self.prompt_model.prepare_model_inputs(batch)
            # check the compatibility for more models. Having checked gpt2, T5
        else: # generating the subsequence generation can use the default setting
            model_inputs = self.plm.prepare_inputs_for_generation(input_ids, **model_kwargs)
        self.last_model_inputs = model_inputs  # to update the model_kwargs in _update_model_kwargs_for_generation, in-place operation.
        return model_inputs


    def _update_model_kwargs_for_generation(self,
        outputs, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:
        r""" The parents class's ``_update_model_kwargs_for_generation`` method will
        add ``past_key_values`` to model_kwargs, and update ``token_type_ids``, and ``attention_mask_ids``.

        In case some of the model_kwargs are modified in the prepare_inputs_for_generation function
        and should be used as the subsequent model_kwargs, we update these kwargs before the parent class
        call.

        Other updates should be added here after the parent's function call.

        Args:
            outputs (:obj:`torch.Tensor`):
            is_encoder_decoder (:obj:`bool`, defaults to False):
        """
        if self.generate_ith_token == 0:
            for key in self.last_model_inputs:
                if key in model_kwargs:
                    model_kwargs[key] = self.last_model_inputs[key]
        model_kwargs = super()._update_model_kwargs_for_generation(outputs=outputs, model_kwargs=model_kwargs, is_encoder_decoder=is_encoder_decoder)
        self.generate_ith_token += 1
        return model_kwargs


    def _prepare_encoder_decoder_kwargs_for_generation(
        self, input_ids: torch.LongTensor, model_kwargs, model_input_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        r""" This function resemble the function in GeneraionMix

        Args:
            input_ids (:obj:`torch.LongTensor`) The input ids for
        """
        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            encoder = self.plm.get_encoder()
            encoder_kwargs = {
                argument: value
                for argument, value in model_kwargs.items()
                if not (argument.startswith("decoder_") or argument.startswith("cross_attn") or argument.startswith("use_cache"))
            }
            model_input_name = model_input_name if model_input_name is not None else self.main_input_name
            batch = {model_input_name:input_ids, **encoder_kwargs}
            model_inputs = self.prompt_model.prepare_model_inputs(batch) # This line differs from the orinigal code base, we should process the input
            # with our template, then pass it into the model.
            # some of the arguments may have been changed by the template,
            # e.g. the attention mask. Here we update the model_kwargs
            for key in model_kwargs:
                if key in model_inputs:
                    model_kwargs[key] = model_inputs[key]
            model_kwargs["encoder_outputs"] = encoder(return_dict=True, **model_inputs)
        return model_kwargs

    ##  We comment this code since it conflict with [OpenDelta](https://github.com/thunlp/OpenDelta)
    # def state_dict(self, *args, **kwargs):
    #     """ Save the model using template and plm's save methods. """
    #     _state_dict = {}
    #     if not self.prompt_model.freeze_plm:
    #         _state_dict['plm'] = self.plm.state_dict(*args, **kwargs)
    #     _state_dict['template'] = self.template.state_dict(*args, **kwargs)
    #     return _state_dict

    # def load_state_dict(self, state_dict, *args, **kwargs):
    #     """ Load the model using template and plm's load methods. """
    #     if 'plm' in state_dict and not self.prompt_model.freeze_plm:
    #         self.plm.load_state_dict(state_dict['plm'], *args, **kwargs)
    #     self.template.load_state_dict(state_dict['template'], *args, **kwargs)

    def _reorder_cache(self, past, beam_idx):
        r"""Use the plm's default _reorder_cache function
        """
        return self.plm._reorder_cache(past, beam_idx)

    def parallelize(self, device_map=None):
        r"""Parallelize the model across device
        """
        if hasattr(self.plm, "parallelize"):
            self.plm.parallelize(device_map)
            self.device_map = self.plm.device_map
        else:
            raise NotImplementedError("parallelize method was not implemented for this plm.")

    def deparallelize(self):
        r"""Deparallelize the model across device
        """
        if hasattr(self.plm, "deparallelize"):
            self.plm.deparallelize()
            self.device_map = None
        else:
            raise NotImplementedError("parallelize method was not implemented for this plm.")
