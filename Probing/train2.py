import pandas as pd
import numpy as np
import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModel

# Ensure proper device is set and free any unnecessary memory
torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Specify which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Load the dataset
dataset_path = 'switched_dataset_3.csv'
dataset = pd.read_csv(dataset_path)

# Split the dataset
train_ratio = 0.8
dev_ratio = 0.2
train_data, dev_data = train_test_split(dataset, train_size=train_ratio, test_size=dev_ratio, random_state=42)

# Load tokenizer and model
model_name = "../models/mt0-xxl-mt"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
# model.to(device)
model.eval()  # Ensure model is in evaluation mode if not training

def get_sentence_embeddings(sentences, batch_size=32):
    embeddings = []
    count=0

    from torch.utils.data import DataLoader
    dataloader = DataLoader(sentences, batch_size=batch_size, shuffle=False)

    for batch_sentences in dataloader:
        count+=1
        inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v for k, v in inputs.items()}

        with torch.no_grad():
            output = model.encoder(**inputs)
            emb = output.last_hidden_state
            sentence_embedding = torch.mean(emb, dim=1)
            embeddings.append(sentence_embedding.cpu())  # Store embeddings on CPU to save GPU memory

        # Delete temporary tensors to free memory
        del inputs, emb, output
        torch.cuda.empty_cache()
        print(f"{count} batches done")

    embeddings = torch.cat(embeddings, dim=0)
    return embeddings

# Extract sentence embeddings
X_train = get_sentence_embeddings(train_data['Sentence'].tolist())
y_train = torch.tensor(train_data['Label'].tolist())

X_dev = get_sentence_embeddings(dev_data['Sentence'].tolist())
y_dev = torch.tensor(dev_data['Label'].tolist())

# MLP model definition
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Hyperparameter search and training
hidden_size_list = [64, 128, 256]
learning_rate_list = [0.001, 0.01, 0.1]
best_accuracy = 0.0
best_hyperparameters = {}

for hidden_size in hidden_size_list:
    for learning_rate in learning_rate_list:
        print(f"Training with hidden_size={hidden_size}, learning_rate={learning_rate}")
        
        mlp_model = MLP(input_size=X_train.shape[1], hidden_size=hidden_size, output_size=1)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(mlp_model.parameters(), lr=learning_rate)

        epochs = 100
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = mlp_model(X_train)
            loss = criterion(outputs, y_train.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            dev_outputs = mlp_model(X_dev)
            predicted_labels = (dev_outputs >= 0.5).squeeze().int()
            accuracy = accuracy_score(y_dev, predicted_labels)
            print(f"Dev accuracy with hidden_size={hidden_size}, learning_rate={learning_rate}: {accuracy * 100}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_hyperparameters = {
                'hidden_size': hidden_size,
                'learning_rate': learning_rate
            }
            print(f"New best hyperparameters found: {best_hyperparameters}")

print("Best hyperparameters:", best_hyperparameters)
print("Best dev accuracy:", best_accuracy * 100)

# Train the final model with the best hyperparameters
final_model = MLP(input_size=X_train.shape[1], hidden_size=best_hyperparameters['hidden_size'], output_size=1)
criterion = nn.BCELoss()
optimizer = optim.Adam(final_model.parameters(), lr=best_hyperparameters['learning_rate'])

epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = final_model(X_train)
    loss = criterion(outputs, y_train.unsqueeze(1).float())
    loss.backward()
    optimizer.step()

# Testing on external data
external_test_data = pd.read_csv('test_sentences.csv')
X_test_external = get_sentence_embeddings(external_test_data['Sentence'].tolist())
y_test_external = torch.tensor(external_test_data['Label'].tolist())

with torch.no_grad():
    test_outputs_external = final_model(X_test_external)
    predicted_labels_external = (test_outputs_external >= 0.5).squeeze().int()
    accuracy_external = accuracy_score(y_test_external, predicted_labels_external)

print("Accuracy on external test set:", accuracy_external * 100)
