from typing import Dict

import torch
from torch.nn import Embedding


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class
        self.embed_dim = embeddings.size(1)

        self.gru = torch.nn.GRU(
            self.embed_dim,
            self.hidden_size, 
            self.num_layers,
            dropout=self.dropout, 
            bidirectional=self.bidirectional, 
            batch_first=True
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.encoder_output_size, self.num_class),
            torch.nn.Softmax()
        )

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        return 2*self.hidden_size*self.num_layers if self.bidirectional else self.hidden_size*self.num_layers

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        inputs = self.embed(batch["text"]).float()
        _, x = self.gru(inputs, None)
        x = torch.permute(x, (1, 0, 2))
        #print(x.size())
        x = torch.reshape(x, (x.size()[0], -1))
        # print(x)
        # print(x.shape)
        x = self.classifier(x)
        return {
          batch["id"][i]: x[i]
          for i in range(len(batch["id"]))
        }


class SeqTagger(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqTagger, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class
        self.embed_dim = embeddings.size(1)

        self.gru = torch.nn.GRU(
            self.embed_dim,
            self.hidden_size, 
            self.num_layers,
            dropout=self.dropout, 
            bidirectional=self.bidirectional, 
            batch_first=True
        )
        self.cnn = []
        for i in range(4):
            self.cnn.append(
                torch.nn.Sequential(
                    torch.nn.ReLU(), 
                    torch.nn.Dropout(dropout), 
                    torch.nn.Conv1d(self.encoder_output_size, self.encoder_output_size, 5, padding = 2)
                )
            )
        self.cnn = torch.nn.ModuleList(self.cnn)
        self.classifier = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.encoder_output_size, self.num_class),
            torch.nn.Sigmoid()
        )
      
    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        return 2*self.hidden_size if self.bidirectional else self.hidden_size

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        inputs = self.embed(batch["tokens"]).float()
        x,_ = self.gru(inputs, None)
        x = torch.permute(x, (0, 2, 1))
        for i in range(4):
            x = self.cnn[i](x)
        x = torch.permute(x, (0, 2, 1))
        x = self.classifier(x)
        return {
            batch["id"][i]: x[i] for i in range(len(batch["id"]))
        }
