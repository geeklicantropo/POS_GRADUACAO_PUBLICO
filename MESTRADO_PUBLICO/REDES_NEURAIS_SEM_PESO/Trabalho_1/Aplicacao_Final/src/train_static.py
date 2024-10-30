import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import numpy as np
from utils import setup_logging, HDF5Dataset
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report
)
import seaborn as sns

# Correctly import torch_dwn
import torch_dwn as dwn

def train_model(classification_type='binary'):
    # Set up logging
    log_file = os.path.join('..', 'logs', f'train_static_{classification_type}.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    setup_logging(log_file)
    logging.info(f"Starting training script for static data ({classification_type} classification)...")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Optional: Set max_split_size_mb to manage large memory allocations
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    # Get the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logging.info(f"Script directory: {script_dir}")

    # Move up one directory to the project root
    project_dir = os.path.dirname(script_dir)
    logging.info(f"Project directory: {project_dir}")

    # Paths for static data
    static_dir = os.path.join(project_dir, 'dados', 'static')
    treated_data_dir = os.path.join(static_dir, 'treated_data', classification_type)
    evaluation_dir = os.path.join(static_dir, 'evaluation', classification_type)
    os.makedirs(evaluation_dir, exist_ok=True)

    # Load class mapping
    class_mapping_path = os.path.join(static_dir, 'processed', f'class_mapping_{classification_type}.pt')
    class_to_idx = torch.load(class_mapping_path)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    # Create datasets
    train_data_file = os.path.join(treated_data_dir, 'train_data.h5')
    val_data_file = os.path.join(treated_data_dir, 'val_data.h5')
    train_dataset = HDF5Dataset(train_data_file)
    val_dataset = HDF5Dataset(val_data_file)

    # Create data loaders with reduced batch size and num_workers
    batch_size = 4  # Further reduced batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Define the model using torch_dwn components
    input_size = train_dataset.data.shape[1]

    if classification_type == 'binary':
        num_output_classes = 2  # For binary classification, we need two output classes
        group_sum_classes = 2
    else:
        num_output_classes = num_classes
        group_sum_classes = num_classes

    # Adjust model parameters to reduce memory usage
    # Change mapping to 'random' to reduce memory consumption
    model = nn.Sequential(
        dwn.LUTLayer(input_size, 1000, n=4, mapping='random'),  # Reduced output_size and n
        dwn.LUTLayer(1000, 500, n=4, mapping='random'),
        dwn.GroupSum(group_sum_classes, tau=1/0.3)
    ).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=14)

    # Training parameters
    num_epochs = 30  # Adjusted to match the example

    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []

        for data, labels in tqdm(train_loader, desc='Training', unit='batch'):
            # Move data and labels to device
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            # Prepare labels
            labels = labels.long()
            # Forward pass
            outputs = model(data)
            # Compute loss
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
            # Compute accuracy and other metrics
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            # Free up memory
            del data, labels, outputs, loss
            torch.cuda.empty_cache()
        epoch_loss = running_loss / len(train_dataset)
        epoch_accuracy = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        logging.info(f"Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        # Adjust learning rate
        scheduler.step()

        # Validation
        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc='Validation', unit='batch'):
                # Move data and labels to device
                data = data.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                labels = labels.long()
                outputs = model(data)
                # Compute loss
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * data.size(0)
                # Compute accuracy and other metrics
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                # Free up memory
                del data, labels, outputs, loss
                torch.cuda.empty_cache()
        val_epoch_loss = val_running_loss / len(val_dataset)
        val_accuracy = correct / total
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_accuracy)
        logging.info(f"Validation Loss: {val_epoch_loss:.4f}, Accuracy: {val_accuracy:.4f}")

    # Save the trained model
    model_path = os.path.join(evaluation_dir, f'static_model_{classification_type}.pt')
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved to {model_path}")

    # Plot and save training and validation metrics
    epochs_list = np.arange(1, num_epochs + 1)

    # Loss curves
    plt.figure()
    plt.plot(epochs_list, train_losses, label='Training Loss')
    plt.plot(epochs_list, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Training and Validation Loss ({classification_type.capitalize()} Classification)')
    loss_plot_path = os.path.join(evaluation_dir, 'loss_curve.png')
    plt.savefig(loss_plot_path)
    plt.close()
    logging.info(f"Loss curve saved to {loss_plot_path}")

    # Accuracy curves
    plt.figure()
    plt.plot(epochs_list, train_accuracies, label='Training Accuracy')
    plt.plot(epochs_list, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'Training and Validation Accuracy ({classification_type.capitalize()} Classification)')
    acc_plot_path = os.path.join(evaluation_dir, 'accuracy_curve.png')
    plt.savefig(acc_plot_path)
    plt.close()
    logging.info(f"Accuracy curve saved to {acc_plot_path}")

    # Classification Report
    class_report = classification_report(all_labels, all_preds, target_names=[idx_to_class[i] for i in range(num_classes)], zero_division=0)
    report_path = os.path.join(evaluation_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(class_report)
    logging.info(f"Classification report saved to {report_path}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[idx_to_class[i] for i in range(num_classes)], yticklabels=[idx_to_class[i] for i in range(num_classes)])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Validation Set ({classification_type.capitalize()} Classification)')
    cm_plot_path = os.path.join(evaluation_dir, 'confusion_matrix.png')
    plt.savefig(cm_plot_path)
    plt.close()
    logging.info(f"Confusion matrix saved to {cm_plot_path}")

    # ROC Curve and AUC (only for binary classification)
    if classification_type == 'binary':
        # Since we have probabilities for both classes, we can calculate ROC AUC
        probs = np.array(all_probs)
        all_labels_np = np.array(all_labels)
        # Use the probability of the positive class (assuming 'malware' is class 1)
        positive_probs = probs[:, 1]
        fpr, tpr, thresholds = roc_curve(all_labels_np, positive_probs)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Validation Set')
        plt.legend()
        roc_plot_path = os.path.join(evaluation_dir, 'roc_curve.png')
        plt.savefig(roc_plot_path)
        plt.close()
        logging.info(f"ROC curve saved to {roc_plot_path}")

    logging.info(f"Training script for static data ({classification_type} classification) completed successfully.")

if __name__ == "__main__":
    # Train for binary classification
    train_model(classification_type='binary')

    # Train for multi-class classification
    train_model(classification_type='multiclass')
