import torch
import tqdm
import json
import os
import argparse
from sklearn.metrics import *
from openprompt.utils.logging import logger
from openprompt.plms import load_plm
from openprompt.prompts import PtuningTemplate, ManualVerbalizer
from openprompt import PromptDataLoader, PromptForClassification
from openprompt.data_utils.text_classification_dataset import CnClickbaitProcessor


# ==================================================================================================
#                                        参数加载函数 (与主脚本一致)
# ==================================================================================================
def load_config_from_file(config_path):
    """从 .txt (JSON格式) 文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    class Config:
        def __init__(self, **entries):
            entries = {k: v for k, v in entries.items() if not k.startswith('//')}
            self.__dict__.update(entries)

    return Config(**config_dict)


# ==================================================================================================
#                                        评估主逻辑
# ==================================================================================================
def evaluate_from_checkpoint(config):
    """根据配置文件，加载模型检查点并对测试集进行评估"""

    logger.info("=" * 20 + " 独立评估脚本 " + "=" * 20)

    # --- 1. 动态构建所有路径 ---
    # 使用相对路径，使其在 Docker 和本地都能正常工作
    project_root = "."  # 当前目录即为项目根目录

    # 从 config 构建模型检查点文件的完整路径
    model_checkpoint_path = os.path.join(project_root, "ckpts", config.final_model_name)

    # 从 config 构建其他所需路径
    model_base_path = os.path.join(project_root, config.model_name_or_path)
    scripts_base_dir = os.path.join(project_root, "scripts", f"TextClassification/{config.dataset}")
    data_path = os.path.join(project_root, "datasets", f"TextClassification/{config.dataset}/")

    logger.info(f"将要评估的模型文件: {model_checkpoint_path}")

    # --- 2. 初始化模型骨架 (与训练时完全一致) ---
    plm, tokenizer, model_config, WrapperClass = load_plm(config.model_type, model_base_path)
    class_labels = CnClickbaitProcessor().get_labels()

    template_path = os.path.join(scripts_base_dir, "ptuning_template.txt")
    with open(template_path, 'r', encoding='utf-8') as f:
        template_text = f.readlines()[config.template_id].rstrip()
    mytemplate = PtuningTemplate(model=plm, tokenizer=tokenizer, text=template_text)

    verbalizer_path = os.path.join(scripts_base_dir, "manual_verbalizer.txt")
    with open(verbalizer_path, 'r', encoding='utf-8') as f:
        verbalizer_dict = json.load(f)
    myverbalizer = ManualVerbalizer(tokenizer, classes=class_labels)
    myverbalizer.label_words = verbalizer_dict['label_words']

    prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)

    # --- 3. 加载训练好的模型权重 ---
    if not os.path.exists(model_checkpoint_path):
        logger.error(f"错误: 模型检查点文件未找到: {model_checkpoint_path}")
        logger.error("请先运行主训练脚本 (fewshot-config.py) 并成功生成模型文件。")
        return

    state_dict = torch.load(model_checkpoint_path,
                            map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    prompt_model.load_state_dict(state_dict)
    logger.info(f"成功从 {model_checkpoint_path} 加载模型权重。")

    if torch.cuda.is_available():
        prompt_model.cuda()

    # --- 4. 加载评估数据 ---
    test_set = CnClickbaitProcessor().get_test_examples(data_path)
    test_dataloader = PromptDataLoader(dataset=test_set, template=mytemplate, tokenizer=tokenizer,
                                       tokenizer_wrapper_class=WrapperClass, max_seq_length=config.max_seq_l,
                                       batch_size=config.batch_size, shuffle=False, truncate_method="tail")
    logger.info(f"已加载 {len(test_set)} 条测试数据进行评估。")

    # --- 5. 执行评估 ---
    prompt_model.eval()
    allpreds = []
    alllabels = []
    with torch.no_grad():
        for inputs in tqdm.tqdm(test_dataloader, desc="Evaluating"):
            if torch.cuda.is_available():
                inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    acc = accuracy_score(alllabels, allpreds)
    pre, recall, f1, _ = precision_recall_fscore_support(alllabels, allpreds, average='macro', zero_division=0)

    # --- 6. 打印评估结果 ---
    report = "=" * 20 + " 评估结果 " + "=" * 20 + "\n"
    report += f"模型: {model_checkpoint_path}\n"
    report += f"数据集: {config.dataset}\n"
    report += "-" * 45 + "\n"
    report += f"Accuracy: {acc:.4f}\n"
    report += f"Precision (Macro): {pre:.4f}\n"
    report += f"Recall (Macro): {recall:.4f}\n"
    report += f"F1-Score (Macro): {f1:.4f}\n"
    report += "=" * 45
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="./config/config.txt")
    args = parser.parse_args()

    # 从主配置文件加载设置
    eval_config = load_config_from_file(args.config_file)

    evaluate_from_checkpoint(eval_config)