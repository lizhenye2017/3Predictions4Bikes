import chardet
import pandas as pd

def convert_csv_encoding(input_file, output_file):
    # 检测文件编码
    with open(input_file, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        original_encoding = result['encoding']
    
    # 如果检测到的编码是GB2312，替换为GBK或GB18030
    if original_encoding.lower() in ['gb2312', 'gbk']:
        original_encoding = 'gb18030'
    
    # 使用检测到的编码读取文件
    try:
        df = pd.read_csv(input_file, encoding=original_encoding)
    except UnicodeDecodeError as e:
        print(f"Error decoding file with {original_encoding}: {e}")
        print("Trying 'latin1' as a fallback encoding.")
        df = pd.read_csv(input_file, encoding='latin1')
    
    # 保存为UTF-8编码的CSV文件
    df.to_csv(output_file, index=False, encoding='utf-8')


input_file = 'data/ori_data/全省实时电价.csv'
output_file = '全省实时电价_utf8.csv'
convert_csv_encoding(input_file, output_file)