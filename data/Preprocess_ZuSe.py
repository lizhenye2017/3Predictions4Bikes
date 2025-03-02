import pandas as pd
from tqdm import tqdm  # 导入tqdm库

def preprocess_csv(file_path, output_file_path=None):
    # 读取CSV文件，显式指定编码为utf-8，分隔符为制表符
    df = pd.read_csv(file_path, encoding='utf-8', delimiter='\t')
    
    # 提取“阻塞”列的不同元素值
    unique_values = df['阻塞'].unique()
    
    # 创建新的DataFrame，包含“时间”列
    new_df = pd.DataFrame()
    new_df['时间'] = df['时间'].drop_duplicates().reset_index(drop=True)
    
    # 为每个阻塞值创建两个新列，并初始化为-1000000
    for val in unique_values:
        new_df[f"{val}_正向极限"] = -1000000
        new_df[f"{val}_反向极限"] = -1000000
    
    # 填充数据，并使用tqdm显示进度条
    for _, row in tqdm(df.iterrows(), total=len(df), desc="处理进度"):
        time = row['时间']
        block = row['阻塞']
        new_df.loc[new_df['时间'] == time, f"{block}_正向极限"] = row['正向极限（MW）']
        new_df.loc[new_df['时间'] == time, f"{block}_反向极限"] = row['反向极限（MW）']
    
    # 如果提供了输出文件路径，则将处理后的数据保存到新文件
    if output_file_path:
        import os
        # 检查并创建目标目录
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        new_df.to_csv(output_file_path, index=False, encoding='utf-8')
        print(f"处理后的数据已保存到 {output_file_path}")
    
    return new_df


# file_path = 'data/ori_data/阻塞信息_utf8.csv'
# output_file_path = 'data/processed_data/阻塞信息_处理后.csv'  # 新文件的保存路径
# result_df = preprocess_csv(file_path, output_file_path)
# print("处理后的数据：")
# print(result_df.head())  # 打印前几行数据以展示处理结果


import pandas as pd
from tqdm import tqdm  # 导入tqdm库

def calculate_limits_sum(file_path, output_file_path):
    # 读取CSV文件，显式指定编码为utf-8，分隔符为逗号
    df = pd.read_csv(file_path, encoding='utf-8')
    
    # 初始化结果DataFrame
    result_df = pd.DataFrame(columns=['时间', '正向极限和', '反向极限和'])
    
    # 遍历每一行数据，并使用tqdm显示进度条
    for _, row in tqdm(df.iterrows(), total=len(df), desc="计算进度"):
        time = row['时间']
        
        # 提取所有正向极限列的值，并过滤掉填充值-1000000
        forward_limits = [row[col] for col in df.columns if col.endswith('_正向极限') and row[col] != -1000000]
        
        # 提取所有反向极限列的值，并过滤掉填充值-1000000
        reverse_limits = [row[col] for col in df.columns if col.endswith('_反向极限') and row[col] != -1000000]
        
        # 如果整行元素都是-1000000，则填入0
        if not forward_limits and not reverse_limits:
            forward_sum = 0
            reverse_sum = 0
        else:
            # 计算正向极限和与反向极限和
            forward_sum = sum(forward_limits)
            reverse_sum = sum(reverse_limits)
        
        # 将结果添加到结果DataFrame中
        new_row = pd.DataFrame({
            '时间': [time],
            '正向极限和': [forward_sum],
            '反向极限和': [reverse_sum]
        })
        result_df = pd.concat([result_df, new_row], ignore_index=True)
    
    # 保存结果到新的CSV文件
    result_df.to_csv(output_file_path, index=False, encoding='utf-8')
    print(f"计算结果已保存到 {output_file_path}")

# 示例用法
file_path = 'data/processed_data/阻塞信息_处理后.csv'  # 输入文件路径
output_file_path = 'data/processed_data/阻塞信息_极限和.csv'  # 新文件的保存路径
calculate_limits_sum(file_path, output_file_path)