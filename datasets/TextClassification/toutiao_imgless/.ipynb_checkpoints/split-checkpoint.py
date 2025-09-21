import pandas as pd
from sklearn.model_selection import train_test_split
import os

# ================================ 配置区域 ================================

# 1. 输入文件：您最终的、完整的、可能包含污染表头的CSV文件
INPUT_CSV = 'toutiao_30.csv' 

# 2. 输出文件路径
TRAIN_CSV_OUTPUT = 'train.csv'
TEST_CSV_OUTPUT = 'test.csv'

# 3. 测试集比例 (0.2 代表 20%)
TEST_SIZE = 0.2

# 4. 随机种子
RANDOM_STATE = 42

# ==============================================================================

def clean_and_split_dataset(input_path, train_path, test_path, test_size, random_state):
    """
    一个集成了清理和分割功能的脚本。
    它会先移除标签不是数字的行，然后对干净的数据进行分层抽样。
    """
    print("--- 开始进行数据集清理与拆分 ---")
    
    if not os.path.exists(input_path):
        print(f"❌ 错误: 输入文件未找到! 路径: {input_path}")
        return

    try:
        df = pd.read_csv(input_path, header=None, names=['label', 'title', 'content', 'image_path'])
        print(f"成功读取 {len(df)} 条总数据。")
    except Exception as e:
        print(f"读取CSV时出错: {e}")
        return

    # --- 步骤 1: 清理数据，移除标签不是数字的行（比如表头行） ---
    original_count = len(df)
    # 将 'label' 列强制转换为数字，无法转换的（如字符串'label'）会变成无效值 (NaN)
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    # 删除包含无效值的行
    df.dropna(subset=['label'], inplace=True)
    # 将标签列安全地转换为整数类型
    df['label'] = df['label'].astype(int)
    
    if original_count > len(df):
        print(f"数据清洗：成功移除了 {original_count - len(df)} 行非数字标签的记录。")
    print(f"剩余有效数据 {len(df)} 条，将对这些数据进行分割。")
    
    # --- 步骤 2: 准备并进行分割 ---
    X = df
    y = df['label']

    print("\n--- 清洗后数据集的类别分布 ---")
    print(y.value_counts(normalize=True))

    print(f"\n正在按 {1-test_size:.0%}:{test_size:.0%} 的比例进行分层抽样...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print("拆分完成。")

    # --- 步骤 3: 保存结果 ---
    print(f"\n正在保存训练集到: {train_path}")
    X_train.to_csv(train_path, index=False, header=False, encoding='utf-8')
    print(f"正在保存测试集到: {test_path}")
    X_test.to_csv(test_path, index=False, header=False, encoding='utf-8')

    print("\n" + "="*50)
    print(" " * 18 + "拆分结果报告")
    print("="*50)
    print(f"有效样本总数: {len(df)}")
    print(f"训练集样本数: {len(X_train)}")
    print(f"测试集样本数: {len(X_test)}")
    print("\n--- 训练集类别分布 ---")
    print(y_train.value_counts(normalize=True))
    print("\n--- 测试集类别分布 ---")
    print(y_test.value_counts(normalize=True))
    print("\n✅ 清理与拆分成功！")
    print("="*50)

if __name__ == "__main__":
    clean_and_split_dataset(
        input_path=INPUT_CSV,
        train_path=TRAIN_CSV_OUTPUT,
        test_path=TEST_CSV_OUTPUT,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )