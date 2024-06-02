import pandas as pd
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
import numpy as np
import os
from sentence_transformers import SentenceTransformer

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# Load the switched dataset
dataset_path = 'dataset.csv'  # Replace with the correct file path
dataset = pd.read_csv(dataset_path)

# Define the train-dev split ratio
train_ratio = 0.8  # 80% for training
dev_ratio = 0.2    # 20% for development (dev)

# Split the dataset into train and dev sets
train_data, dev_data = train_test_split(dataset, train_size=train_ratio, test_size=dev_ratio, random_state=42)

# Load the pre-trained MT5 model and tokenizer
model_name = "../models/mt0-xl"
# model_st = SentenceTransformer(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to get sentence embeddings from the model
def get_sentence_embeddings(sentences):
    embeddings = []
    count=0
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
        output = model.encoder(
                    input_ids=inputs["input_ids"], 
                    attention_mask=inputs["attention_mask"], 
                    return_dict=True
                )   
        # get the final hidden states
        emb = output.last_hidden_state
        sentence_embedding = torch.mean(emb, dim=1)
        embeddings.append(sentence_embedding)
        count+=1
        if(count%10==0):
            print(f"{count} instances done")
    return torch.tensor(embeddings)

# Get sentence embeddings for train and dev data
X_train = get_sentence_embeddings(train_data['Sentence'].tolist())
y_train = torch.tensor(train_data['Label'].tolist())
X_dev = get_sentence_embeddings(dev_data['Sentence'].tolist())
y_dev = torch.tensor(dev_data['Label'].tolist())

# Define and train the MLP model with hyperparameter tuning
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

# Define hyperparameters to tune (e.g., hidden_size, learning_rate, etc.)
hidden_size_list = [64, 128, 256]
learning_rate_list = [0.001, 0.01, 0.1]

best_accuracy = 0.0
best_hyperparameters = {}

for hidden_size in hidden_size_list:
    for learning_rate in learning_rate_list:
        mlp_model = MLP(input_size=X_train.shape[1], hidden_size=hidden_size, output_size=1)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(mlp_model.parameters(), lr=learning_rate)

        # Train the model on the train set
        epochs = 10
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = mlp_model(X_train)
            loss = criterion(outputs, y_train.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

        # Evaluate the model on the dev set
        with torch.no_grad():
            dev_outputs = mlp_model(X_dev)
            predicted_labels = (dev_outputs >= 0.5).squeeze().int()
            accuracy = accuracy_score(y_dev, predicted_labels)

        # Update best hyperparameters if the accuracy is higher
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_hyperparameters = {
                'hidden_size': hidden_size,
                'learning_rate': learning_rate
            }

print("Best hyperparameters:", best_hyperparameters)
print("Best dev accuracy:", best_accuracy * 100)

# Train the final model with the best hyperparameters on the entire train dataset
final_model = MLP(input_size=X_train.shape[1], hidden_size=best_hyperparameters['hidden_size'], output_size=1)
criterion = nn.BCELoss()
optimizer = optim.Adam(final_model.parameters(), lr=best_hyperparameters['learning_rate'])

epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = final_model(X_train)
    loss = criterion(outputs, y_train.unsqueeze(1).float())
    loss.backward()
    optimizer.step()

# Now, you can use the trained final model for testing on your external test set
# Load and preprocess your external test data similarly as done for the train and dev data
# Then, evaluate the model using the test data

# Example:
external_test_data = pd.read_csv('test_sentences.csv')
X_test_external = get_sentence_embeddings(external_test_data['Sentence'].tolist())
y_test_external = torch.tensor(external_test_data['Label'].tolist())

# Test the final model on the external test set
with torch.no_grad():
    test_outputs_external = final_model(X_test_external)
    predicted_labels_external = (test_outputs_external >= 0.5).squeeze().int()
    accuracy_external = accuracy_score(y_test_external, predicted_labels_external)

print("Accuracy on external test set:", accuracy_external * 100)