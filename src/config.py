from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
LOGS_DIR = ROOT_DIR / 'logs'
MODELS_DIR = ROOT_DIR / 'models'


# 超参数
SEQ_LEN = 32
EMBEDDING_DIM = 128
ENCODE_HIDDEN_SIZE = 256
# 为什么*2 因为encode 输出的hidden 是双向的 拼接起来之后 的hidden size 就是 2 * encode_hidden_size
DECODE_HIDDEN_SIZE = ENCODE_HIDDEN_SIZE *2
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001
ENCODER_LAYERS = 2