import torch.nn as nn

class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x hidden_size
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output

class MDLSTM(nn.Module):
    # The visual features of textline are given as input to a MDLSTM
    # Each of the LSTMs then recursively maps these features into a lower dimensional space
    # The standard one dimensional LSTM network can be extended to multiple dimensions by using n self connections with n forget gates
    # Inspired by HM-LSTM originally proposed in - https://arxiv.org/pdf/1609.01704.pdf
    def __init__(self, input_size, hidden_size, output_size):
        super(MDLSTM, self).__init__()
        self.rnn = nn.Sequential(
                LSTM(input_size, hidden_size, 2*hidden_size),
                LSTM(2*hidden_size, hidden_size, 4*hidden_size),
                LSTM(4*hidden_size, hidden_size, 2*hidden_size),
                LSTM(2*hidden_size, hidden_size, hidden_size))
        self.linear = nn.Linear(hidden_size, output_size)
    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        for rnn in self.rnn:
            rnn.rnn.flatten_parameters()
        recurrent = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x hidden_size
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output

# import torch
# x = torch.randn(1,100, 512)
# net1 = BidirectionalLSTM(512, 256, 512)
# net2 = LSTM(512, 256, 512)
# net3 = GRU(512, 256, 512)
# net4 = MDLSTM(512, 256, 512)

# print("=========================================")
# out1 = net1(x)
# print(out1.shape)
# print("=========================================")
# out2 = net2(x)
# print(out2.shape)
# print("=========================================")
# out3 = net3(x)
# print(out3.shape)
# print("=========================================")
# out4 = net4(x)
# print(out4.shape)