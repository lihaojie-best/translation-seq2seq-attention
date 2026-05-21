import time
from itertools import chain

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import config
from src.dataset import get_dataloader
from src.model import TranslationEncoder, TranslationDecoder
from tokenizer import ChineseTokenizer, EnglishTokenizer


def train_one_epoch(encoder, decoder, dataloader, loss_function, optimizer, device):
    # 设为训练模式
    encoder.train()
    decoder.train()
    epoch_total_loss = 0  # 总损失
    for inputs, targets in tqdm(dataloader, desc='训练'):
        # 数据移动到设备
        inputs = inputs.to(device)
        targets = targets.to(device)  # targets.shape [batch_size, seq_len] seq_len= <sos>*<eos>
        # 梯度清0
        optimizer.zero_grad()
        # 编码
        context_vector = encoder(inputs)  # context_vector.shape [batch_size, encode_hidden_size * 2]
        # 解码
        decoder_input = targets[:, 0:1]  # decoder_input.shape [batch_size, <sos>]
        decoder_hidden = context_vector.unsqueeze(0)  # decoder_hidden.shape [1, batch_size, encode_hidden_size * 2]
        decoder_outputs = []  # decoder_outputs.shape [batch_size, <sos>, vocab_size=seq_len-1]
        # 循环编码
        for t in range(1, targets.shape[1]):  # targets.shape[1] = seq_len -1
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            # 收集预测结果
            decoder_outputs.append(decoder_output)
            # 修改下一个循环的输入
            decoder_input = targets[:, t:t + 1]
        # 拼接预测结果
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = decoder_outputs.reshape(-1, decoder_outputs.shape[
            -1])  # decoder_outputs.shape[-1] =最后一个维度的大小 也就是 词表大小
        # 期望值
        decoder_targets = targets[:, 1:]  # decoder_targets.shape [batch_size, seq_len-1]
        decoder_targets = decoder_targets.reshape(-1)  # decoder_targets.shape [batch_size * seq_len-1]
        # 计算损失
        loss = loss_function(decoder_outputs, decoder_targets)
        loss.backward()
        optimizer.step()
        epoch_total_loss += loss.item()
    return epoch_total_loss / len(dataloader)


def train():
    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 准备数据集
    chinese_tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DATA_DIR / "cn_vocab.txt")
    english_tokenizer = EnglishTokenizer.from_vocab(config.PROCESSED_DATA_DIR / "en_vocab.txt")
    # 准备模型
    encoder = TranslationEncoder(vocab_size=chinese_tokenizer.vocab_size,
                                 padding_idx=chinese_tokenizer.pad_token_id).to(device)
    decoder = TranslationDecoder(vocab_size=english_tokenizer.vocab_size,
                                 padding_idx=english_tokenizer.pad_token_id).to(device)

    # 加载数据
    dataloader = get_dataloader()
    # 准备损失函数
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=english_tokenizer.pad_token_id)
    # 优化器
    optimizer = torch.optim.Adam(params=chain(encoder.parameters(), decoder.parameters()), lr=config.LEARNING_RATE)
    # tensorboard
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    # 训练
    best_loss = float('inf')
    for epoch in range(1, config.EPOCHS + 1):
        print(f"========Epoch:{epoch}========")
        avg_loss = train_one_epoch(encoder, decoder, dataloader, loss_function, optimizer, device)
        print(f"Loss:{avg_loss:.4f}")
        writer.add_scalar("Loss", avg_loss, epoch)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(encoder.state_dict(), config.MODELS_DIR / "encoder.pt")
            torch.save(decoder.state_dict(), config.MODELS_DIR / "decoder.pt")
            print("模型保存成功")
        else:
            print("模型保存失败")


if __name__ == '__main__':
    train()
