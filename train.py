import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from model import SimpleCNN
from dataset import CustomImageDataset

def train_batch(model, images, labels, criterion, optimizer, device):
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss, outputs

def compute_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    predicted = predicted.to(labels.device)
    correct_predictions = (predicted == labels).sum().item()
    return correct_predictions

def log_epoch_metrics(total_loss, total_correct, total_samples, epoch, global_epoch, train_loader):
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * total_correct / total_samples
    print(f"Epoch [{epoch + 1}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

def perform_epoch(model, train_loader, criterion, optimizer, device, epoch, global_epoch):
    total_loss = 0
    total_correct = 0
    total_samples = 0

    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        loss, outputs = train_batch(model, images, labels, criterion, optimizer, device)
        total_loss += loss.item()
        total_correct += compute_accuracy(outputs, labels)
        total_samples += labels.size(0)

    log_epoch_metrics(total_loss, total_correct, total_samples, epoch, global_epoch, train_loader)

def validate_model(model, val_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += compute_accuracy(outputs, labels)
            total += labels.size(0)
    return 100 * correct / total

def train_fold(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, patience, output_name, fold, global_epoch):
    best_val_accuracy = 0
    early_stop_counter = 0

    for epoch in range(num_epochs):
        perform_epoch(model, train_loader, criterion, optimizer, device, epoch, global_epoch)
        
        val_accuracy = validate_model(model, val_loader, device)
        print(f"Epoch [{epoch + 1}], Fold {fold + 1} Validation Accuracy: {val_accuracy:.2f}%")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            early_stop_counter = 0
            # torch.save(model.state_dict(), output_name)
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        global_epoch += 1

    return global_epoch

def train_cross_validation(model, dataset, device, output_name, k_folds=5, num_epochs=10, patience=1):
    kfold = KFold(n_splits=k_folds, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    global_epoch = 0

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold + 1}/{k_folds}")
        train_loader, val_loader = get_data_loaders(dataset, train_ids, val_ids)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        global_epoch = train_fold(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, patience, output_name, fold, global_epoch)

def get_data_loaders(dataset, train_ids, val_ids, batch_size=100):
    train_subsampler = Subset(dataset, train_ids)
    val_subsampler = Subset(dataset, val_ids)
    train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def setup_and_train_model(model, output_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    print("Loading dataset")
    dataset = CustomImageDataset(root_dir="testdata")

    print("Training with cross-validation")
    train_cross_validation(model, dataset, device, output_name, k_folds=5, num_epochs=10)

def main():
    setup_and_train_model(SimpleCNN(), "model.pth")

if __name__ == '__main__':
    main()
