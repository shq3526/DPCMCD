
import copy
import json
import pickle
from typing import *

import torch
from torch.utils.data._utils.collate import default_collate
from openprompt.utils.logging import logger

from typing import Union


class InputExample(object):
    """A raw input example consisting of segments of text,
    a label for classification task or a target sequence of generation task.
    Other desired information can be passed via meta.

    Args:
        guid (:obj:`str`, optional): A unique identifier of the example.
        text_a (:obj:`str`, optional): The placeholder for sequence of text.
        text_b (:obj:`str`, optional): A secend sequence of text, which is not always necessary.
        label (:obj:`int`, optional): The label id of the example in classification task.
        tgt_text (:obj:`Union[str,List[str]]`, optional):  The target sequence of the example in a generation task..
        meta (:obj:`Dict`, optional): An optional dictionary to store arbitrary extra information for the example.
    """

    def __init__(self,
                 guid = None,
                 text_a = "",
                 text_b = "",
                 label = None,
                 meta: Optional[Dict] = None,
                 tgt_text: Optional[Union[str,List[str]]] = None
                ):

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.meta = meta if meta else {}
        self.tgt_text = tgt_text

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        r"""Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        r"""Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    def keys(self, keep_none=False):
        return [key for key in self.__dict__.keys() if getattr(self, key) is not None]

    @staticmethod
    def load_examples(path: str) -> List['InputExample']:
        """Load a set of input examples from a file"""
        with open(path, 'rb') as fh:
            return pickle.load(fh)

    @staticmethod
    def save_examples(examples: List['InputExample'], path: str) -> None:
        """Save a set of input examples to a file"""
        with open(path, 'wb') as fh:
            pickle.dump(examples, fh)


# openprompt/data_utils/utils.py

class InputFeatures(dict):
    """
    The class for input to the PLM and Prompts. To make users explicitly know the available keys,
    we define a dict with a set of predefined possible keys. The default value to any key is None.
    When use it as a dict, all the keys whose values are None are invisible.
    This class support most of the dict's operation (See Examples). It can also be consumed by
    pytorch's default_collate in DataLoader.
    Also a :py:meth:`to_tensor()` method is build to convert the values into torch.Tensor for torch's input.
    """
    tensorable_keys = ['input_ids', 'inputs_embeds', 'attention_mask', 'token_type_ids', 'label',
        'decoder_input_ids', 'decoder_inputs_embeds', 'soft_token_ids',
        'past_key_values', 'loss_ids', 'image_features']
    all_keys = ['input_ids', 'inputs_embeds', 'attention_mask', 'token_type_ids', 'label',
        'decoder_input_ids', 'decoder_inputs_embeds', 'soft_token_ids',
        'past_key_values', 'loss_ids', 'guid', 'tgt_text', 'encoded_tgt_text', 'input_ids_len','meta', 'title', 'image_features']
    non_tensorable_keys = []

    def __init__(self,
                input_ids: Optional[Union[List, torch.Tensor]] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                attention_mask: Optional[Union[List[int], torch.Tensor]] = None,
                token_type_ids: Optional[Union[List[int], torch.Tensor]] = None,
                label: Optional[Union[int, torch.Tensor]] = None,
                decoder_input_ids: Optional[Union[List, torch.Tensor]] = None,
                decoder_inputs_embeds: Optional[torch.Tensor] = None,
                soft_token_ids: Optional[Union[List, torch.Tensor]] = None,
                past_key_values: Optional[torch.Tensor] = None,
                loss_ids: Optional[Union[List, torch.Tensor]] = None,
                guid: Optional[str] = None,
                tgt_text: Optional[str] = None,
                use_cache: Optional[bool] = None,
                encoded_tgt_text: Optional[str] = None,
                input_ids_len: Optional[int] = None,
                meta: Optional[Union[List, torch.Tensor]] = None,
                title: Optional[str] = None,
                image_features: Optional[torch.Tensor] = None,
                **kwargs):

        self.input_ids = input_ids
        self.inputs_embeds = inputs_embeds
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.decoder_input_ids = decoder_input_ids
        self.decoder_inputs_embeds = decoder_inputs_embeds
        self.soft_token_ids = soft_token_ids
        self.past_key_values = past_key_values
        self.loss_ids = loss_ids
        self.guid = guid
        self.tgt_text = tgt_text
        self.encoded_tgt_text = encoded_tgt_text
        self.use_cache = use_cache
        self.input_ids_len = input_ids_len
        self.meta = meta
        self.title = title
        self.image_features = image_features
        for k in kwargs.keys():
            logger.warning("Your are passing an unexpected key words: {} to InputFeatures, might yield unexpected behaviours!".format(k))
            setattr(self, k, kwargs[k])

    @classmethod
    def add_tensorable_keys(cls, *args):
        cls.tensorable_keys.extend(args)

    @classmethod
    def add_not_tensorable_keys(cls, *args):
        cls.not_tensorable_keys.extend(args)

    @classmethod
    def add_keys(cls, *args):
        cls.all_keys.extend(args)

    def __repr__(self):
        return str(self.to_json_string())

    def __len__(self):
        return len(self.keys())

    def to_tensor(self, device: str = 'cuda'):
        """inplace operation, convert all tensorable features into :obj:`torch.tensor`"""
        for key in self.tensorable_keys:
            value = getattr(self, key)
            if value is not None and not isinstance(value, torch.Tensor):
                setattr(self, key, torch.tensor(value))
        return self

    def to(self, device: str = "cuda:0"):
        r"""move the tensor keys to runtime device, such as gpu:0
        """
        for key in self.tensorable_keys:
            value = getattr(self, key)
            if value is not None and isinstance(value, torch.Tensor):
                setattr(self, key, value.to(device))
        return self

    def cuda(self, device: str = "cuda:0"):
        r"""mimic the tensor behavior
        """
        return self.to(device)

    def to_json_string(self, keep_none=False):
        """Serializes this instance to a JSON string."""
        data = {}
        for key in self.all_keys:
            value = getattr(self, key)
            if isinstance(value, torch.Tensor):
                data[key] =  value.detach().cpu().tolist()
            elif value is None and keep_none:
                data[key] = None
            else:
                data[key] = value
        return json.dumps(data) + "\n"

    def keys(self, keep_none=False) -> List[str]:
        if keep_none:
            return self.all_keys
        else:
            return [key for key in self.all_keys if getattr(self, key) is not None]

    def to_dict(self, keep_none=False) -> Dict[str, Any]:
        data = {}
        for key in self.all_keys:
            value = getattr(self, key)
            if value is not None:
                data[key] =  value
            elif value is None and keep_none:
                data[key] = None
        return data

    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        return iter(self.keys())

    def __setitem__(self, key, item):
        if key not in self.all_keys:
            raise KeyError("Key {} not in predefined set of keys".format(key))
        setattr(self, key, item)

    def values(self, keep_none=False) -> List[Any]:
        return [getattr(self, key) for key in self.keys(keep_none=keep_none)]

    def __contains__(self, key, keep_none=False):
        return key in self.keys(keep_none)

    def items(self,):
        return [(key, self.__getitem__(key)) for key in self.keys()]

    @staticmethod
    def collate_fct(batch: List):
        elem = batch[0]
        return_dict = {}
        for key in elem:
            if key == "encoded_tgt_text":
                return_dict[key] = [d[key] for d in batch]
            else:
                try:
                    return_dict[key] = default_collate([d[key] for d in batch])
                except:
                    print(f"key{key}\n d {[batch[i][key] for i in range(len(batch))]} ")

        return InputFeatures(**return_dict)