import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

# ----------------------------
# 1. Configuración
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = 10
seq_length = 5
embedding_dim = 16
hidden_dim = 32
num_epochs = 1000
batch_size = 64

# ----------------------------
# 2. Dataset sintético
# ----------------------------
def generate_batch(batch_size, seq_length, vocab_size):
    X = np.random.randint(1, vocab_size, (batch_size, seq_length))
    Y = X.copy()
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)

# ----------------------------
# 3. Atención (Bahdanau simple)
# ----------------------------
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.size(1)
        print(src_len)
        # repetir hidden por src_len para poder concatenar
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # calcular "energías"
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)

# ----------------------------
# 4. Encoder
# ----------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        emb = self.embedding(x)
        outputs, (h, c) = self.lstm(emb)
        return outputs, (h, c)  # outputs = todos los pasos

# ----------------------------
# 5. Decoder con Atención
# ----------------------------
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, attention):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        self.attention = attention

    def forward(self, x, hidden, cell, encoder_outputs):
        # x: (batch, 1)
        emb = self.embedding(x)  # (batch, 1, emb_dim)
        hidden_last = hidden[-1]  # (batch, hidden_dim)

        # Calcular pesos de atención
        attn_weights = self.attention(hidden_last, encoder_outputs)  # (batch, src_len)

        # Aplicar atención: contexto = suma ponderada de encoder_outputs
        attn_weights = attn_weights.unsqueeze(1)  # (batch, 1, src_len)
        context = torch.bmm(attn_weights, encoder_outputs)  # (batch, 1, hidden_dim)

        # Concatenar contexto con la entrada embebida
        rnn_input = torch.cat((emb, context), dim=2)

        outputs, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        prediction = self.fc(torch.cat((outputs.squeeze(1), context.squeeze(1)), dim=1))

        return prediction.unsqueeze(1), hidden, cell, attn_weights

# ----------------------------
# 6. Seq2Seq (modelo completo)
# ----------------------------
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.shape
        vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(device)

        encoder_outputs, (h, c) = self.encoder(src)
        input = trg[:, 0].unsqueeze(1)  # primer token

        for t in range(1, trg_len):
            output, h, c, attn = self.decoder(input, h, c, encoder_outputs)
            outputs[:, t] = output.squeeze(1)
            top1 = output.argmax(2)
            input = trg[:, t].unsqueeze(1) if random.random() < teacher_forcing_ratio else top1

        return outputs

# ----------------------------
# 7. Entrenamiento
# ----------------------------
attn = Attention(hidden_dim)
encoder = Encoder(vocab_size, embedding_dim, hidden_dim)
decoder = Decoder(vocab_size, embedding_dim, hidden_dim, attn)
model = Seq2Seq(encoder, decoder).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

for epoch in range(num_epochs):
    X, Y = generate_batch(batch_size, seq_length, vocab_size)
    X, Y = X.to(device), Y.to(device)

    optimizer.zero_grad()
    output = model(X, Y)
    loss = criterion(output[:, 1:].reshape(-1, vocab_size), Y[:, 1:].reshape(-1))
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# ----------------------------
# 8. Evaluación
# ----------------------------
model.eval()
with torch.no_grad():
    X_test, Y_test = generate_batch(5, seq_length, vocab_size)
    X_test = X_test.to(device)
    output = model(X_test, Y_test, teacher_forcing_ratio=0.0)
    preds = output.argmax(2).cpu().numpy()

    for i in range(5):
        print(f"\nInput:    {X_test[i].cpu().numpy()}")
        print(f"Predicho: {preds[i]}")
        print(f"Esperado: {Y_test[i].cpu().numpy()}")
