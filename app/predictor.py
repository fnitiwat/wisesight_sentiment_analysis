import torch
import os
import numpy as np
from typing import List
from pythainlp.tokenize import word_tokenize
from torch import nn


class LSTMNet(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim=300, hidden_dim=256):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])


class MySentimentModel:
    def __init__(self, model_path, vocab_path, max_lenght, class_names, device):
        self.vocab = torch.load(vocab_path)
        self.net = LSTMNet(
            vocab_size=len(self.vocab),
            output_size=len(class_names),
        )
        if device=="cpu":
         self.net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        else:
            self.net.load_state_dict(torch.load(model_path))
        self.max_lenght = max_lenght
        self.class_names = class_names
        self.device = device
        self.net.eval().to(self.device)

    def predict(self, text: str) -> str:
        batch = self._preprocess(text)
        logits = self.net.forward(batch)
        class_name = self.class_names[np.argmax(logits.detach().cpu().numpy())]
        return class_name

    def _preprocess(self, text: str):
        # tokenize
        tokenized_words = self._tokenize(text)
        # encode
        encoded_words = self._encode(tokenized_words)
        # padding
        encoded_words = self._padding(encoded_words)
        # expand to batch size = 1
        batch = torch.LongTensor([encoded_words]).to(self.device)
        return batch

    def _tokenize(self, text: str) -> List[str]:
        tokenized_words = word_tokenize(text=text)
        return tokenized_words

    def _encode(self, tokenized_words: List[str]) -> List[int]:
        return numericalize_data(tokenized_words, self.vocab)

    def _padding(self, encoded_words: List[int]) -> List[int]:
        features = np.ones(self.max_lenght, dtype=int) * self.vocab["<pad>"]
        features[-len(encoded_words) :] = np.array(encoded_words)[: self.max_lenght]
        return features.tolist()


def numericalize_data(tokens, vocab):
    ids = [vocab[token] for token in tokens]
    return ids


class Config:
    def __init__(self):
        self.model_path = "../artifacts/model.pth"
        self.vocab_path = "../artifacts/vocab.pth"
        self.max_lenght = 748
        self.class_names = ["neg", "neu", "pos", "q"]
        self.device = os.getenv("DEVICE", "cuda")

config = Config()

predictor = MySentimentModel(
    model_path=config.model_path,
    vocab_path=config.vocab_path,
    max_lenght=config.max_lenght,
    class_names=config.class_names,
    device=config.device
)
