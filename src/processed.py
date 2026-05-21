import pandas as pd
from sklearn.model_selection import train_test_split

import config
from tokenizer import ChineseTokenizer, EnglishTokenizer


def process():
    print("数据处理开始")
    # 读取数据
    df = pd.read_csv(
        config.RAW_DATA_DIR / "cmn.txt", sep="\t",
        header=None,  # 数据集是tsv 格式 没有头
        usecols=[0, 1],  # 没有头也就没有列，所以指定列序号
        names=["en", "cn"],  # 后面处理 使用0，1 列名语意不明确 因此添加列名
        encoding="utf-8"
    )
    # 数据清洗
    df = df.dropna()
    df = df[  # pandas 里的 「行过滤器」
        df["en"].str.strip().ne("") & df["cn"].str.strip().ne("")  # ne = not equal（不等于） ne("")不等于空字符串
        ]
    # 划分数据集
    train_df, test_df = train_test_split(df, test_size=0.2)
    # 构建中英文 两个词表 (基于训练集）
    ChineseTokenizer.build_vocab(train_df["cn"].to_list(), config.PROCESSED_DATA_DIR / "cn_vocab.txt")
    EnglishTokenizer.build_vocab(train_df["en"].to_list(), config.PROCESSED_DATA_DIR / "en_vocab.txt")
    # 从已保存的文件中构建Tokenizer 对象 用来处理训练集与数据集
    chinese_tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DATA_DIR / "cn_vocab.txt")
    english_tokenizer = EnglishTokenizer.from_vocab(config.PROCESSED_DATA_DIR / "en_vocab.txt")
    # 计算sel_len 也就是token长度 需要分词之后
    # cn_len = train_df["cn"].apply(lambda x: len(chinese_tokenizer.tokenize(x))).max()
    # en_len = train_df["en"].apply(lambda x: len(english_tokenizer.tokenize(x))).max()
    # print("cn_len:", cn_len)
    # print("en_len:", en_len)
    # 构建训练集并保存
    train_df['cn']=train_df['cn'].apply(lambda x: chinese_tokenizer.encode(x, seq_len=config.SEQ_LEN, add_sos_eos=False))
    train_df['en']=train_df['en'].apply(lambda x: english_tokenizer.encode(x, seq_len=config.SEQ_LEN, add_sos_eos=True))
    train_df.to_json(config.PROCESSED_DATA_DIR / "indexed_train.jsonl", orient='records', lines=True)
    # 构建训练集并保存
    test_df['cn']=test_df['cn'].apply(lambda x: chinese_tokenizer.encode(x, seq_len=config.SEQ_LEN, add_sos_eos=False))
    test_df['en']=test_df['en'].apply(lambda x: english_tokenizer.encode(x, seq_len=config.SEQ_LEN, add_sos_eos=True))
    test_df.to_json(config.PROCESSED_DATA_DIR / "indexed_test.jsonl", orient='records', lines=True)
    print("数据预处理结束")
if __name__ == '__main__':
    process()
