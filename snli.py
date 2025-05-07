import pandas as pd

data = pd.read_json("snli_1.0/snli_1.0_train.jsonl", lines = True)
print(data.head())

X = [data["sentence1"][i]+" implies "+data["sentence2"][i] for i in range(550152)]

y = []
for i in range(550152):
    if data["gold_label"][i] == "neutral":
        y.append(0)
    elif data["gold_label"][i] == "entailment":
        y.append(1)
    else:
        y.append(2)
        
        
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)


#########
# 2. Konwertuj tekst na indeksy wg słownika GloVe
def text_to_indices(text):
    return [glove.stoi.get(token, 0) for token in tokenizer(text)]  # 0 = indeks dla nieznanych słów


# 4. Własny dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.data = [torch.tensor(text_to_indices(t), dtype=torch.long) for t in texts]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = TextDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=1000, shuffle=True, collate_fn=collate_batch)


# 5. collate_fn do obsługi paddingu i długości
def collate_batch(batch):
    sequences, labels = zip(*batch)

    # Sortuj malejąco po długości (wymagane przez pack_padded_sequence)
    seq_lens = [len(seq) for seq in sequences]
    sorted_indices = sorted(range(len(seq_lens)), key=lambda i: -seq_lens[i])
    sequences = [sequences[i] for i in sorted_indices]
    labels = torch.tensor([labels[i] for i in sorted_indices])

    # Padujemy sekwencje do najdłuższej
    padded_seqs = pad_sequence(sequences, batch_first=True)  # [batch, max_seq_len]

    lengths = torch.tensor([len(seq) for seq in sequences])  # rzeczywiste długości

    return padded_seqs, lengths, labels

# 6. DataLoader z naszym collate_fn

# 7. Klasyfikator tekstu z LSTM i GloVe
class T_LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(glove.vectors, freeze=False)  # Wbudowane embeddingi GloVe
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        # x: [batch_size, max_len], lengths: [batch_size]
        embedded = self.embedding(x)  # [batch, max_len, emb_dim]

        # Pakujemy sekwencje, ignorując paddingi
        packed_input = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=True)

        # Przechodzimy przez LSTM
        packed_output, (hn, cn) = self.lstm(packed_input)

        # hn: [num_layers * num_directions, batch, hidden_size]
        last_hidden = hn[-1]  # jeśli jednokierunkowy LSTM, to bierzemy ostatnią warstwę

        # Klasyfikacja (na podstawie ostatniego hidden state)
        output = self.fc(last_hidden)  # [batch, num_classes]
        return output

# 8. Inicjalizacja modelu i optymalizatora
embedding_dim = glove.dim  # 100
hidden_size = 30
num_classes = 3

model = T_LSTM(embedding_dim, hidden_size, num_classes)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


###############################################
from sklearn.metrics import accuracy_score, classification_report

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in loader:
        inputs, lengths, targets = batch
        outputs = model(inputs, lengths)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.tolist())
        all_labels.extend(targets.tolist())

# Accuracy
acc = accuracy_score(all_labels, all_preds)
print(f"\n🔍 Accuracy: {acc:.4f}")

# Szczegółowy raport (precision, recall, f1)
print("\n📊 Classification report:")
print(classification_report(all_labels, all_preds))


##################################################


print("Rozpoczynam trening")
# 9. Trening
for epoch in range(10):
    model.train()
    total_loss = 0
    for batch in loader:
        inputs, lengths, targets = batch

        optimizer.zero_grad()
        outputs = model(inputs, lengths)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")
    
    
#########################################

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in loader:
        inputs, lengths, targets = batch
        outputs = model(inputs, lengths)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.tolist())
        all_labels.extend(targets.tolist())

# Accuracy
acc = accuracy_score(all_labels, all_preds)
print(f"\n🔍 Accuracy: {acc:.4f}")

# Szczegółowy raport (precision, recall, f1)
print("\n📊 Classification report:")
print(classification_report(all_labels, all_preds))


##########
print("zbiot testowy")


dataset2 = TextDataset(X_test, y_test)
loader2 = DataLoader(dataset2, batch_size=1000, shuffle=True, collate_fn=collate_batch)


model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in loader2:
        inputs, lengths, targets = batch
        outputs = model(inputs, lengths)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.tolist())
        all_labels.extend(targets.tolist())

# Accuracy
acc = accuracy_score(all_labels, all_preds)
print(f"\n🔍 Accuracy: {acc:.4f}")

# Szczegółowy raport (precision, recall, f1)
print("\n📊 Classification report:")
print(classification_report(all_labels, all_preds))