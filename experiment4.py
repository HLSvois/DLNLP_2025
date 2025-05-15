import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 超参数
SEQ_LEN = 100
BATCH_SIZE = 512
LR = 1e-3
DEVICE = 'cuda:2' if torch.cuda.is_available() else 'mps'
TEMPERATURES = [0.7, 1.0, 1.3]
EPOCHS_LIST = [1, 2, 5, 10, 20]
TRANSFORMER_LAYERS = [2, 4, 6]
LOG_FILE = 'train_log.txt'

os.makedirs("checkpoints", exist_ok=True)

def log(text):
    with open(LOG_FILE, 'a') as f:
        print(text)
        f.write(text + '\n')

def load_data(directory):
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='GB18030') as f:
                texts.append(f.read())
    return '\n'.join(texts)

text = load_data('/data/luhao/HW4/jyxstxtqj_downcc')
paragraphs = text.split('\n')
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])
data = [torch.tensor(encode(paragraph), dtype=torch.long) for paragraph in paragraphs]

class CharDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
        self.items = self._prepare_data()

    def _prepare_data(self):
        items = []
        for paragraph in self.data:
            for i in range(len(paragraph) - self.seq_len):
                input_seq = paragraph[i:i + self.seq_len]
                target_seq = paragraph[i + 1:i + 1 + self.seq_len]
                items.append((input_seq, target_seq))
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

dataset = CharDataset(data, SEQ_LEN)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, nhead=4, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x) * (x.size(1) ** 0.5)
        x = x.permute(1, 0, 2)
        out = self.transformer(x)
        out = out.permute(1, 0, 2)
        return self.fc(out)

def generate_text(model, start_str='张无忌', length=200, temperature=1.0):
    model.eval()
    input_ids = torch.tensor(encode(start_str), dtype=torch.long).unsqueeze(0).to(DEVICE)
    generated = input_ids.tolist()[0]
    hidden = None
    for _ in range(length):
        if isinstance(model, LSTMModel):
            output, hidden = model(input_ids, hidden)
        else:
            output = model(input_ids)
        next_logits = output[:, -1, :] / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_id)
        input_ids = torch.tensor([[next_id]], dtype=torch.long).to(DEVICE)
    return decode(generated)

def train_model(model, model_name, epochs):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            output, *_ = model(x) if isinstance(model, LSTMModel) else (model(x),)
            loss = criterion(output.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        log(f"{model_name} Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")
    ckpt_path = f"checkpoints/{model_name}_epoch{epochs}.pt"
    torch.save(model.state_dict(), ckpt_path)
    log(f"Saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    for epochs in EPOCHS_LIST:
        # LSTM 部分
        lstm_model = LSTMModel(vocab_size)
        lstm_name = f"LSTM_e{epochs}"
        train_model(lstm_model, lstm_name, epochs)
        for T in TEMPERATURES:
            text = generate_text(lstm_model, temperature=T)
            log(f"[{lstm_name}] Temperature={T}\n{text}\n")

        # Transformer 部分
        for nl in TRANSFORMER_LAYERS:
            trans_model = TransformerModel(vocab_size, num_layers=nl)
            trans_name = f"Transformer_L{nl}_e{epochs}"
            train_model(trans_model, trans_name, epochs)
            for T in TEMPERATURES:
                text = generate_text(trans_model, temperature=T)
                log(f"[{trans_name}] Temperature={T}\n{text}\n")
