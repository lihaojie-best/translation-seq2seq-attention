import torch
import config
from src.model import TranslationEncoder, TranslationDecoder
from src.tokenizer import ChineseTokenizer, EnglishTokenizer


def predict_batch(input_tensor, encoder, decoder, chinese_tokenizer, english_tokenizer, device):
    # 模型开启评估模式
    encoder.eval()
    decoder.eval()
    # 取消梯度计算
    with torch.no_grad():
        # 编码
        context_vector = encoder(input_tensor)
        # context_vector.shape: [batch_size, decoder_hidden_size]

        # 解码
        batch_size = input_tensor.shape[0]
        decoder_input = torch.full((batch_size, 1), english_tokenizer.sos_token_id, device=device)
        # decoder_input.shape: [batch_size, 1]

        decoder_hidden = context_vector.unsqueeze(0)
        # decoder_hidden.shape: [1, batch_size, decoder_hidden_size]

        generated = [[] for _ in range(batch_size)]
        is_finished = [False for _ in range(batch_size)]
        for t in range(config.SEQ_LEN):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            # decoder_output.shape: [batch_size, 1, vocab_size]
            predict_indexes = torch.argmax(decoder_output, dim=-1)
            # predict_indexes.shape: [batch_size, 1]

            # 处理每个时间步的预测结果
            for i in range(batch_size):
                if is_finished[i]:
                    continue
                else:
                    if predict_indexes[i].item() == english_tokenizer.eos_token_id:
                        is_finished[i] = True
                    else:
                        generated[i].append(predict_indexes[i].item())

            if all(is_finished):
                break
            decoder_input = predict_indexes
        return generated


def predict(user_input, encoder, decoder, chinese_tokenizer, english_tokenizer, device):
    # 处理数据
    index_list = chinese_tokenizer.encode(user_input, config.SEQ_LEN, add_sos_eos=False)
    input_tensor = torch.tensor(index_list).to(device)
    batch_result = predict_batch(input_tensor, encoder, decoder, chinese_tokenizer, english_tokenizer, device)
    result = batch_result[0]
    return english_tokenizer.decode(result)


def run_predict():
    # 准备资源
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    # 加载 tokenizer
    chinese_tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DATA_DIR / "cn_vocab.txt")
    english_tokenizer = EnglishTokenizer.from_vocab(config.PROCESSED_DATA_DIR / "en_vocab.txt")
    # 加载模型
    encoder = TranslationEncoder(vocab_size=chinese_tokenizer.vocab_size,
                                 padding_idx=chinese_tokenizer.pad_token_id).to(device)
    encoder.load_state_dict(torch.load(config.MODELS_DIR / "encoder.pt"))
    decoder = TranslationDecoder(vocab_size=english_tokenizer.vocab_size,
                                 padding_idx=english_tokenizer.pad_token_id).to(device)
    decoder.load_state_dict(torch.load(config.MODELS_DIR / "decoder.pt"))
    # 运行预测
    print('中英翻译：（输入q或者quit退出）')
    while True:
        user_input = input('中文：')
        if user_input in ['q', 'quit']:
            break
        if user_input.strip() == '':
            print('请输入内容')
            continue
        result = predict(user_input, encoder, decoder, chinese_tokenizer, english_tokenizer, device)
        print('英文：', result)


if __name__ == '__main__':
    run_predict()
