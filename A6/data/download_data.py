import os
import json


def process_jsonl(input_file, de_output_file, en_output_file):
    """处理jsonl文件并分别保存德语和英语文本"""
    de_texts = []
    en_texts = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            de_texts.append(data['de'])
            en_texts.append(data['en'])
    
    # 保存德语文件
    with open(de_output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(de_texts))
    
    # 保存英语文件
    with open(en_output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(en_texts))


def main():
    # 获取数据目录
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    # 处理每个数据集分割
    for split in ['train', 'test', 'val']:
        print(f"处理{split}数据集...")
        
        # 设置输入和输出文件路径
        input_file = os.path.join(data_dir, f"{split}.jsonl")
        de_output = os.path.join(data_dir, f"{split}.de")
        en_output = os.path.join(data_dir, f"{split}.en")
        
        # 处理数据
        process_jsonl(input_file, de_output, en_output)
        print(f"{split}数据集已处理完成")
    
    print("所有数据集处理完成！")


if __name__ == '__main__':
    main()
