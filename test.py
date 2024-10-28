import torch
from torch.utils.data import DataLoader

from model import SimpleCNN
from dataset import CustomImageDataset

# Function to evaluate the model on test data and calculate accuracy
def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Calculate accuracy if labels are available
            if labels[0] != -1:
                labels = labels.to(device)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

    accuracy = 100 * correct / total if total > 0 else None
    
    return accuracy

def setup_and_test_model(model, model_path):
    test_data_dir = "./testdata"

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the test dataset
    print('Loading dataset')
    test_dataset = CustomImageDataset(root_dir=test_data_dir)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load the model
    print('Loading model ' + model_path)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully.")

    # Evaluate the model
    print('Testing model ' + model_path)
    accuracy = evaluate_model(model, test_loader, device)

    # Display accuracy if calculated
    if accuracy is not None:
        print(f"\tAccuracy: {accuracy:.2f}%")

def main():
    setup_and_test_model(SimpleCNN(), 'model.pth')

if __name__ == "__main__":
    main()
