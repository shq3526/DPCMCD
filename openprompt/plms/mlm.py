
from transformers.models.auto.tokenization_auto import tokenizer_class_from_name

from openprompt.plms.utils import TokenizerWrapper
from typing import List, Dict
from collections import defaultdict

class MLMTokenizerWrapper(TokenizerWrapper):
    add_input_keys = ['input_ids', 'attention_mask', 'token_type_ids']

    @property
    def mask_token(self):
        return self.tokenizer.mask_token

    @property
    def mask_token_ids(self):
        return self.tokenizer.mask_token_id

    @property
    def num_special_tokens_to_add(self):
        if not hasattr(self, '_num_specials'):
            self._num_specials = self.tokenizer.num_special_tokens_to_add()
        return self._num_specials

    def tokenize_one_example(self, wrapped_example, teacher_forcing):
        ''' # TODO doesn't consider the situation that input has two parts
        '''

        wrapped_example, others = wrapped_example

        encoded_tgt_text = []
        if 'tgt_text' in others:
            tgt_text = others['tgt_text']
            if isinstance(tgt_text, str):
                tgt_text = [tgt_text]
            for t in tgt_text:
                encoded_tgt_text.append(self.tokenizer.encode(t, add_special_tokens=False))

        mask_id = 0

        encoder_inputs = defaultdict(list)
        for piece in wrapped_example:

            # ======================== 核心修复逻辑开始 ========================
            #
            # 我们在这里增加一个“净化”步骤，确保 piece['text'] 永远不会是 None。
            # 1. 安全地获取 'text' 的值
            text_to_process = piece.get('text')
            # 2. 如果值为 None，则将其转换为空字符串 ""
            if text_to_process is None:
                text_to_process = ""
            #
            # 在本循环的后续部分，我们将使用 text_to_process 而不是 piece['text']
            #
            # ======================== 核心修复逻辑结束 ========================

            if piece['loss_ids'] == 1:
                if teacher_forcing:
                    raise RuntimeError("Masked Language Model can't perform teacher forcing training!")
                else:
                    encode_text = [self.mask_token_ids]
                mask_id += 1

            # (已修改) 使用净化后的 text_to_process
            if text_to_process in self.special_tokens_maps.keys():
                to_replace = self.special_tokens_maps[text_to_process]
                if to_replace is not None:
                    # 注意：这里我们仍然修改原始的 piece['text']，因为后续逻辑可能依赖它
                    piece['text'] = to_replace
                    text_to_process = to_replace  # 同时更新净化后的变量
                else:
                    raise KeyError("This tokenizer doesn't specify {} token.".format(text_to_process))

            if 'soft_token_ids' in piece and piece['soft_token_ids'] != 0:
                encode_text = [0]
            else:
                # (已修改) 使用净化后的 text_to_process，确保不会传入 None
                encode_text = self.tokenizer.encode(text_to_process, add_special_tokens=False)

            encoding_length = len(encode_text)
            encoder_inputs['input_ids'].append(encode_text)
            for key in piece:
                if key not in ['text']:
                    encoder_inputs[key].append([piece[key]] * encoding_length)

        encoder_inputs = self.truncate(encoder_inputs=encoder_inputs)
        encoder_inputs.pop("shortenable_ids")
        encoder_inputs = self.concate_parts(input_dict=encoder_inputs)
        encoder_inputs = self.add_special_tokens(encoder_inputs=encoder_inputs)
        encoder_inputs['attention_mask'] = [1] * len(encoder_inputs['input_ids'])
        if self.create_token_type_ids:
            encoder_inputs['token_type_ids'] = [0] * len(encoder_inputs['input_ids'])

        encoder_inputs = self.padding(input_dict=encoder_inputs, max_len=self.max_seq_length,
                                      pad_id_for_inputs=self.tokenizer.pad_token_id)

        if len(encoded_tgt_text) > 0:
            encoder_inputs = {**encoder_inputs, "encoded_tgt_text": encoded_tgt_text}
        else:
            encoder_inputs = {**encoder_inputs}
        return encoder_inputs










