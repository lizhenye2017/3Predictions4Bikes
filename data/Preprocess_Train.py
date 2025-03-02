import pandas as pd
import os
from tqdm import tqdm  # 导入tqdm库

def generate_quarterly_times(start_time, end_time):
    # 生成一刻钟间隔的时间序列
    return pd.date_range(start=start_time, end=end_time, freq='15T')

def fill_hourly_data(df, time_col):
    # 将小时数据填充到一刻钟间隔
    df[time_col] = pd.to_datetime(df[time_col])  # 确保时间列为datetime类型
    df.set_index(time_col, inplace=True)
    df = df.resample('15T').ffill()  # 向前填充
    df.reset_index(inplace=True)
    return df

def merge_csv_files(file_paths, output_file_path, start_time, end_time):
    # 生成一刻钟间隔的时间序列
    quarterly_times = generate_quarterly_times(start_time, end_time)
    merged_df = pd.DataFrame({ '时间': quarterly_times })

    # 使用tqdm显示进度条
    for file_path in tqdm(file_paths, desc="合并进度"):
        # 读取CSV文件
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # 判断时间间隔是否为小时
        time_diff = pd.to_datetime(df['时间']).diff().min()
        if time_diff == pd.Timedelta(hours=1):
            # 如果是小时数据，填充到一刻钟间隔
            df = fill_hourly_data(df, '时间')
        else:
            # 如果是一刻钟数据，直接保留
            df['时间'] = pd.to_datetime(df['时间'])
        
        # 将当前文件的数据合并到merged_df中
        merged_df = merged_df.merge(df, on='时间', how='left')

    # 保存合并后的数据到新的CSV文件
    merged_df.to_csv(output_file_path, index=False, encoding='utf-8')
    print(f"合并后的数据已保存到 {output_file_path}")

# 示例用法
file_paths = [
    'data/processed_data/信息合并_A.csv',  # 以小时为间隔的文件
    'data/ori_data/全省实时电价_utf8.csv'   # 以一刻钟为间隔的文件
]
output_file_path = 'data/processed_data/信息合并_B.csv'  # 新文件的保存路径
start_time = '2023-10-01 00:15:00+08'  # 起始时间
end_time = '2024-12-01 00:00:00+08'    # 结束时间
merge_csv_files(file_paths, output_file_path, start_time, end_time)