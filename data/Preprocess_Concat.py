import pandas as pd
import os
from tqdm import tqdm  # 导入tqdm库

def merge_csv_files(file_paths, output_file_path):
    # 初始化一个空的DataFrame，用于存储合并后的数据
    merged_df = pd.DataFrame()

    # 使用tqdm显示进度条
    for file_path in tqdm(file_paths, desc="合并进度"):
        # 读取CSV文件
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # 设置“时间”列为索引
        df.set_index('时间', inplace=True)
        
        # 将当前文件的数据合并到merged_df中
        if merged_df.empty:
            merged_df = df
        else:
            merged_df = merged_df.join(df, how='outer')  # 使用outer join确保所有时间都被保留

    # 重置索引，将“时间”列恢复为普通列
    merged_df.reset_index(inplace=True)

    # 保存合并后的数据到新的CSV文件
    merged_df.to_csv(output_file_path, index=False, encoding='utf-8')
    print(f"合并后的数据已保存到 {output_file_path}")

# 示例用法
file_paths = [
    'data/ori_data/备用信息_utf8.csv',
    'data/ori_data/负荷信息_utf8.csv',
    'data/ori_data/联络线信息_utf8.csv',
    'data/ori_data/阻塞信息_极限和.csv'
]
output_file_path = 'data/processed_data/信息合并.csv'  # 新文件的保存路径
merge_csv_files(file_paths, output_file_path)
#