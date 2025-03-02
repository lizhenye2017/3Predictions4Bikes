import pandas as pd

def extract_columns(input_file_path, output_file_path, columns_to_extract):
    """
    从CSV文件中提取指定列并保存到新的CSV文件中

    参数:
    - input_file_path: 输入CSV文件路径
    - output_file_path: 输出CSV文件路径
    - columns_to_extract: 需要提取的列名列表
    """
    # 读取CSV文件
    df = pd.read_csv(input_file_path, encoding='utf-8')
    
    # 检查指定的列是否存在
    missing_columns = [col for col in columns_to_extract if col not in df.columns]
    if missing_columns:
        raise ValueError(f"以下列名不存在于文件中: {missing_columns}")
    
    # 提取指定列
    extracted_df = df[columns_to_extract]
    
    # 保存到新的CSV文件
    extracted_df.to_csv(output_file_path, index=False, encoding='utf-8')
    print(f"提取的列已保存到 {output_file_path}")

# 示例用法
input_file_path = 'data/processed_data/train_data.csv'  # 输入文件路径
output_file_path = 'data/processed_data/Extract_train_data_A.csv'  # 输出文件路径
columns_to_extract = ['时间','最小正备用（MW）','最大正备用','最大负备用（MW）']  # 需要提取的列名
extract_columns(input_file_path, output_file_path, columns_to_extract)

#