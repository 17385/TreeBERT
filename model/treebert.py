import torch.nn as nn

class TreeBERT(nn.Module):
    def __init__(self, transformer, vocab_size):
        super().__init__()
        self.transformer = transformer
        self.mask_lm = MaskedLanguageModel(self.transformer.hidden, vocab_size)
        self.node_order_prediction = NodeOrderPrediction(self.transformer.hidden)

    def forward(self, x, p, y):
        output = self.transformer(x,p,y)
        return self.node_order_prediction(output[:,-2:-1,:]), self.mask_lm(output)

class NodeOrderPrediction(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.linear(x).squeeze(1)
        return x

class MaskedLanguageModel(nn.Module):
    def __init__(self, hidden, vocab_size):
        super().__init__()
        self.fc_out = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.fc_out(x)
        return self.softmax(x)
