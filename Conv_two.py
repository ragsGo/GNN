import pandas as pd
import numpy as np
import torch
from torchtext import data
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import from_networkx, from_scipy_sparse_matrix
from scipy import sparse
filename = 'csv-data/WHEAT_combined.csv'

with open(filename) as fp:
    line = fp.readline()
    column_count = len(line.split(","))
value_columns = [str(( i +1)) for i in range(column_count -1)]
labels = ["value"] + value_columns
df_whole = pd.read_csv(filename, names=labels)

train, test_df = train_test_split(df_whole, test_size=0.2)
train_df, valid_df = train_test_split(train, test_size = 0.1)


BATCH_SIZE = 12

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# x_train = torch.tensor(train_df.values.tolist(), dtype=torch.float)
x_train = train_df.loc[:, train_df.columns != 'value']
y_train = train_df['value']
x_test = test_df.loc[:, test_df.columns != 'value']
y_test = test_df['value']
x_valid = valid_df.loc[:, valid_df.columns != 'value']
y_valid  = valid_df['value']
# x_valid = torch.tensor(valid_df.values.tolist(), dtype=torch.float)
# x_test = torch.tensor(test_df.values.tolist(), dtype=torch.float)


class CNN(nn.Module):
    def __init__(self, inp_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()

        self.embedding = nn.Embedding(inp_size, embedding_dim)
        self.conv_0 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_sizes[0], embedding_dim))
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_sizes[1], embedding_dim))
        self.conv_2 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_sizes[2], embedding_dim))
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [sent len, batch size]

        #         x = x.permute(1, 0)

        # x = [batch size, sent len]

        embedded = self.embedding(x)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))

        # conv_n = [batch size, n_filters, sent len - filter_sizes[n]]

        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)

INPUT_DIM = len(train_df)
EMBEDDING_DIM = 100
N_FILTERS = 1000
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 1
DROPOUT = 0.5

model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
# pretrained_embeddings =x_train.values
# print(type(pretrained_embeddings))
# model.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings).float() )
import torch.optim as optim

optimizer = optim.Adam(model.parameters())

criterion = torch.nn.MSELoss()

model = model.to(device)
criterion = criterion.to(device)


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.text).squeeze(1)

        loss = criterion(predictions, batch.label)

        #acc = binary_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, batch.label)

            epoch_loss += loss.item()


    return epoch_loss / len(iterator)

N_EPOCHS = 5

for epoch in range(N_EPOCHS):
    train_loss = train(model, x_train, optimizer, criterion)
    valid_loss = evaluate(model, x_valid, criterion)

    print(
        f'| Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f}  Val. Loss: {valid_loss:.3f} ')