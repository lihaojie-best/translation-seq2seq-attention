import torch
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

from tokenizer import ChineseTokenizer, EnglishTokenizer
import config
from model import TranslationEncoder, TranslationDecoder
from dataset import get_dataloader
from predict import predict_batch


def evaluate(dataloader, encoder, decoder, zh_tokenizer, en_tokenizer, device):
    references = []  # [[[4,5,6,7]],[[5,6,7,8,9]],[[7,8,9]]]
    predictions = []  # [[4,5,6,7],[5,6,7,8,9],[7,8,9]]

    special_tokens = [en_tokenizer.sos_token_id, en_tokenizer.eos_token_id, en_tokenizer.pad_token_id]

    for inputs, targets in tqdm(dataloader, desc='评估'):
        inputs = inputs.to(device)
        # inputs.shape: [batch_size, seq_len]

        targets = targets.tolist()
        # 参考译文: [[*,*,*,*,*,*],[*,*,*,*,*,*],[*,*,*,*,*,*]]

        batch_result = predict_batch(inputs, encoder, decoder, zh_tokenizer, en_tokenizer, device)
        # 预测译文：batch_result.shape: [[4,6,7],[11,23,45,78,99],[88,99,26,55]...]

        # 处理预测结果
        predictions.extend(batch_result)

        # 获取参考译文
        references.extend([[[index for index in target if index not in special_tokens]] for target in targets])

    return corpus_bleu(references, predictions)


def run_evaluate():
    # 准备资源
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # tokenizer
    cn_tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'cn_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'en_vocab.txt')

    # 模型
    encoder = TranslationEncoder(vocab_size=cn_tokenizer.vocab_size,
                                 padding_idx=cn_tokenizer.pad_token_id).to(device)
    encoder.load_state_dict(torch.load(config.MODELS_DIR / 'encoder.pt'))

    decoder = TranslationDecoder(vocab_size=en_tokenizer.vocab_size,
                                 padding_idx=en_tokenizer.pad_token_id).to(device)
    decoder.load_state_dict(torch.load(config.MODELS_DIR / 'decoder.pt'))

    # 加载数据集
    dataloader = get_dataloader(train=False)

    bleu = evaluate(dataloader, encoder, decoder, cn_tokenizer, en_tokenizer, device)

    print(f'Bleu: {bleu}')


if __name__ == '__main__':
    run_evaluate()
