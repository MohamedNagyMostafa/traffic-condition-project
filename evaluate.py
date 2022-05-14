import torch

class LSTMClassifier(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn1 = nn.LSTM(input_dim, hidden_dim, layer_dim, bidirectional=True, batch_first=True)
        self.rnn2 = nn.LSTM(hidden_dim * 2, hidden_dim, layer_dim, bidirectional=True, batch_first=True)
        #self.rnn3 = nn.LSTM(hidden_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.batch_size = None
        self.hidden = None

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (h1, c1) = self.rnn1(x, (h0, c0))
        out, (h2, c2) = self.rnn2(out, (h1, c1))
        #out, (_, _) = self.rnn3(out, (h2, c2))
        #print('out: ', out.shape)
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]