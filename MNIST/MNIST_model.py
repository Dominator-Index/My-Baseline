# model.py
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from MNIST_data_utils import generate_target_matrix # Assuming data_utils.py is the source
import wandb

# Removed balanced_init as it was commented out and not used.

class LinearNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device):
        super(LinearNetwork, self).__init__()
        # Using nn.Linear handles default PyTorch initialization (Uniform) 
        self.fc1 = nn.Linear(input_dim, hidden_dim).to(device)
        self.act1 = nn.Tanh() # Tanh activation specified
        self.fc2 = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.act2 = nn.Tanh() # Tanh activation specified 
        self.fc3 = nn.Linear(hidden_dim, output_dim).to(device)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x # (batch_size, output_dim)

# Modified test_model to use MSE loss and correct label types
def test_model(model, X_test, Y_test_labels, Y_test_onehot, device):
    model.eval()  
    # Use MSELoss for testing as well, consistent with training setup 
    loss_fn = nn.MSELoss()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        y_predict = model.forward(X_test)
        # For MSE loss, the target is Y_test_onehot
        loss = loss_fn(y_predict, Y_test_onehot)
        total_loss += loss.item() * X_test.size(0) 
        total_samples += X_test.size(0)

        _, predicted_indices = torch.max(y_predict, 1)
        # Ensure comparison is done on the same device
        predicted_indices = predicted_indices.to(device)
        correct_predictions += (predicted_indices == Y_test_labels).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    # Logging test loss and accuracy
    wandb.log({"test_loss": avg_loss, "test_accuracy": accuracy})
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")