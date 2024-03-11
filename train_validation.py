import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, random_split
import numpy as np

from proccesing_dataset import UrbanSoundDataset
from cnn import CNNetwork

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = "./data/metadata/UrbanSound8K.csv"
AUDIO_DIR = "./data/audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

def create_data_loaders(dataset, batch_size, split_ratio=[0.8, 0.1, 0.1]):
    # Calculate lengths of splits
    train_len = int(len(dataset) * split_ratio[0])
    val_len = int(len(dataset) * split_ratio[1])
    test_len = len(dataset) - train_len - val_len
    
    # Split dataset
    train_data, val_data, test_data = random_split(dataset, [train_len, val_len, test_len])
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)
        # forward pass
        prediction = model(input)
        loss = loss_fn(prediction, target)
        
        # backward pass and optimization
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
        # calculate metrics
        running_loss += loss.item() * input.size(0)
        _, predicted = torch.max(prediction, 1)
        correct_predictions += (predicted == target).sum().item()
        total_samples += target.size(0)
    
    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples
    return epoch_loss, epoch_accuracy

def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for input, target in data_loader:
            input, target = input.to(device), target.to(device)
            # forward pass
            prediction = model(input)
            loss = loss_fn(prediction, target)
            
            # calculate metrics
            running_loss += loss.item() * input.size(0)
            _, predicted = torch.max(prediction, 1)
            correct_predictions += (predicted == target).sum().item()
            total_samples += target.size(0)
    
    loss = running_loss / total_samples
    accuracy = correct_predictions / total_samples
    return loss, accuracy

def train(model, train_loader, val_loader, test_loader, loss_fn, optimiser, device, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}:")
        
        train_loss, train_accuracy = train_single_epoch(model, train_loader, loss_fn, optimiser, device)
        print(f"  Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)
        print(f"  Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        
        print("-" * 50)
    
    test_loss, test_accuracy = evaluate(model, test_loader, loss_fn, device)
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate dataset object
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    dataset = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)
    
    # Split dataset into train, validation, and test sets
    train_loader, val_loader, test_loader = create_data_loaders(dataset, BATCH_SIZE)
    
    # Initialize model, loss function, and optimizer
    model = CNNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    train(model, train_loader, val_loader, test_loader, loss_fn, optimiser, device, EPOCHS)

    # Save the trained model
    torch.save(model.state_dict(), "CNNetwork.pth")
    print("Model Trained and saved as CNNNnetwork.pth")
