import os
import time  
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  
import logging
import matplotlib.pyplot as plt
import numpy as np
from utils import setup_logging
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
import seaborn as sns
import pandas as pd  

import torch_dwn as dwn

# ---------------------------------------------------------------------
# Custom Dataset Class for Dynamic Malware Data
# ---------------------------------------------------------------------
class DynamicDataset(Dataset):
    """
    Custom Dataset class for dynamic malware data.

    Attributes:
        data (torch.Tensor): Feature tensor containing the dynamic data.
        labels (torch.Tensor): Tensor containing the labels.
    """
    def __init__(self, data_dict):
        """
        Initializes the DynamicDataset with data and labels.

        Args:
            data_dict (dict): Dictionary containing 'data' and 'labels' keys.
        """
        self.data = data_dict['data']
        self.labels = torch.tensor(data_dict['labels']).long()

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.labels.size(0)

    def __getitem__(self, idx):
        """
        Retrieves the data and label at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (data, label) pair.
        """
        return self.data[idx], self.labels[idx]

# ---------------------------------------------------------------------
# Function to Train the Model
# ---------------------------------------------------------------------
def train_model(classification_type='binary', num_epochs=30):
    """
    Trains a Discriminative Weight Network (DWN) model for dynamic malware detection.

    This function performs a single run of training and inference, records relevant metrics,
    and generates comprehensive reports and visualizations.

    Args:
        classification_type (str): 'binary' for binary classification.
        num_epochs (int): Number of epochs to train the model.
    """
    # -----------------------------------------------------------------
    # Initialization and Setup
    # -----------------------------------------------------------------
    #Record the start time of the entire training process
    start_time = time.time()

    #Set up logging
    log_file = os.path.join('..', 'logs', f'train_dynamic_{classification_type}.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    setup_logging(log_file)
    logging.info(f"Starting training script for dynamic data ({classification_type} classification)...")

    #Set the device to GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    #Set max_split_size_mb to manage large memory allocations
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    #Get the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logging.info(f"Script directory: {script_dir}")

    #Move up one directory to the project root
    project_dir = os.path.dirname(script_dir)
    logging.info(f"Project directory: {project_dir}")

    #Paths for dynamic data
    dynamic_dir = os.path.join(project_dir, 'dados', 'dynamic')
    treated_data_dir = os.path.join(dynamic_dir, 'treated_data')
    evaluation_dir = os.path.join(dynamic_dir, 'evaluation', classification_type)
    os.makedirs(evaluation_dir, exist_ok=True)  #Create evaluation directory if it doesn't exist

    #Load training and validation data
    train_data_file = os.path.join(treated_data_dir, 'train_data.pt')
    val_data_file = os.path.join(treated_data_dir, 'val_data.pt')

    #Check if training and validation data files exist
    if not os.path.exists(train_data_file):
        logging.error(f"Training data file not found at {train_data_file}")
        return
    if not os.path.exists(val_data_file):
        logging.error(f"Validation data file not found at {val_data_file}")
        return

    #Load data using torch.load
    train_data = torch.load(train_data_file)
    val_data = torch.load(val_data_file)
    logging.info("Training and validation data loaded successfully.")

    #Create training and validation datasets
    train_dataset = DynamicDataset(train_data)
    val_dataset = DynamicDataset(val_data)
    logging.info("Training and validation datasets created.")

    #Create data loaders with reduced batch size and num_workers
    batch_size = 4  
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    logging.info(f"Data loaders created with batch size {batch_size}.")

    #Define the input size based on dataset
    input_size = train_dataset.data.shape[1]
    num_classes = 2  #Binary classification (Goodware vs Malware)

    # -----------------------------------------------------------------
    # Model, Loss Function, Optimizer, and Scheduler Setup
    # -----------------------------------------------------------------
    #Initialize the model
    model = nn.Sequential(
        #First LUTLayer with random mapping to reduce memory usage
        dwn.LUTLayer(input_size, 1000, n=4, mapping='random'),
        #Second LUTLayer with random mapping
        dwn.LUTLayer(1000, 500, n=4, mapping='random'),
        #GroupSum layer for classification
        dwn.GroupSum(num_classes, tau=1/0.3)
    ).to(device)
    logging.info("Model initialized and moved to device.")

    #Define loss function (CrossEntropyLoss for classification tasks)
    criterion = nn.CrossEntropyLoss()
    logging.info("Loss function (CrossEntropyLoss) defined.")

    #Define optimizer (Adam optimizer with learning rate 1e-2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    logging.info("Optimizer (Adam) initialized with learning rate 1e-2.")

    #Learning rate scheduler to adjust the learning rate during training
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=14)
    logging.info("Learning rate scheduler (StepLR) initialized with gamma=0.1 and step_size=14.")

    # -----------------------------------------------------------------
    # Initialize Lists to Store Metrics
    # -----------------------------------------------------------------
    train_losses = []        #List to store training loss per epoch
    val_losses = []          #List to store validation loss per epoch
    train_accuracies = []    #List to store training accuracy per epoch
    val_accuracies = []      #List to store validation accuracy per epoch

    all_labels = []          #List to store true labels during classification
    all_preds = []           #List to store predicted labels during classification
    all_probs = []           #List to store predicted probabilities during classification

    # -----------------------------------------------------------------
    # Training Loop
    # -----------------------------------------------------------------
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch+1}/{num_epochs}")
        epoch_start_time = time.time()  #Record start time of the epoch

        model.train()  #Set model to training mode
        running_loss = 0.0  #Accumulate training loss
        correct = 0          #Number of correct predictions
        total = 0            #Total number of samples

        #Iterate over batches in the training loader
        for data, labels in tqdm(train_loader, desc='Training', unit='batch'):
            #Move data and labels to the device
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()  #Zero the parameter gradients

            labels = labels.long()  #Ensure labels are of type long

            outputs = model(data)  #Forward pass

            loss = criterion(outputs, labels)  #Compute loss

            loss.backward()  #Backward pass
            optimizer.step()  #Update weights

            running_loss += loss.item() * data.size(0)  #Update running loss

            #Compute predictions
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)  #Update total samples
            correct += (predicted == labels).sum().item()  #Update correct predictions

            #Free up memory
            del data, labels, outputs, loss
            torch.cuda.empty_cache()

        #Compute average loss and accuracy for the epoch
        epoch_loss = running_loss / len(train_dataset)
        epoch_accuracy = correct / total

        train_losses.append(epoch_loss)         #Store training loss
        train_accuracies.append(epoch_accuracy) #Store training accuracy

        logging.info(f"Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}")

        #Step the scheduler to adjust learning rate
        scheduler.step()

        # -----------------------------------------------------------------
        # Validation Loop
        # -----------------------------------------------------------------
        model.eval()  #Set model to evaluation mode
        val_running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            #Iterate over batches in the validation loader
            for data, labels in tqdm(val_loader, desc='Validation', unit='batch'):
                #Move data and labels to the device
                data = data.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                labels = labels.long()

                outputs = model(data)  #Forward pass

                loss = criterion(outputs, labels)  #Compute loss

                val_running_loss += loss.item() * data.size(0)  #Update validation loss

                #Compute predictions
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                #Free up memory
                del data, labels, outputs, loss
                torch.cuda.empty_cache()

        #Compute average validation loss and accuracy
        val_epoch_loss = val_running_loss / len(val_dataset)
        val_accuracy = correct / total

        val_losses.append(val_epoch_loss)         #Store validation loss
        val_accuracies.append(val_accuracy)       #Store validation accuracy

        logging.info(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        #Log time taken for the epoch
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        logging.info(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds.")

    # -----------------------------------------------------------------
    # End of Training Loop
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # Classification (Inference) Phase
    # -----------------------------------------------------------------
    classification_start_time = time.time()  #Record start time of classification

    model.eval()  #Ensure model is in evaluation mode

    with torch.no_grad():
        #Iterate over batches in the validation loader
        for data, labels in tqdm(val_loader, desc='Classification', unit='batch'):
            #Move data and labels to the device
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(data)  #Forward pass

            #Compute probabilities and predictions
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            #Append labels, predictions, and probabilities for metrics
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            #Free up memory
            del data, labels, outputs
            torch.cuda.empty_cache()

    #Record classification time
    classification_end_time = time.time()
    classification_time = classification_end_time - classification_start_time

    #Total time (training + classification)
    total_time = classification_end_time - start_time

    logging.info(f"Classification Time: {classification_time:.2f} seconds")
    logging.info(f"Total Time (Training + Classification): {total_time/60:.2f} minutes")

    # -----------------------------------------------------------------
    # Metrics Calculation
    # -----------------------------------------------------------------
    #Compute precision, recall, and F1-score
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0)

    #Compute ROC AUC
    try:
        roc_auc = roc_auc_score(all_labels, [prob[1] for prob in all_probs])
    except ValueError:
        roc_auc = float('nan')  #ROC AUC is not defined in some cases. Why? I have no ideia

    # -----------------------------------------------------------------
    # Save Metrics Summary
    # -----------------------------------------------------------------
    # Create a DataFrame for metrics summary
    metrics_summary = pd.DataFrame({
        'Metric': ['F1-Score', 'Precision', 'Recall', 'Accuracy', 'ROC AUC', 
                   'Training Time (s)', 'Classification Time (s)', 'Total Time (s)'],
        'Value': [
            f1_score,
            precision,
            recall,
            val_accuracies[-1],
            roc_auc,
            training_time := classification_start_time - start_time,  #Define training_time
            classification_time,
            total_time
        ]
    })
    logging.info("Metrics summary created.")

    #Save metrics summary to CSV
    metrics_summary_path = os.path.join(evaluation_dir, f'{classification_type}_metrics_summary.csv')
    metrics_summary.to_csv(metrics_summary_path, index=False)
    logging.info(f"Metrics summary saved to {metrics_summary_path}")

    # -----------------------------------------------------------------
    # Plot and Save Metrics Curves
    # -----------------------------------------------------------------
    epochs_list = np.arange(1, num_epochs + 1)

    #Plot and save training and validation loss curves
    plt.figure()
    plt.plot(epochs_list, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs_list, val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    loss_plot_path = os.path.join(evaluation_dir, 'loss_curve.png')
    plt.savefig(loss_plot_path)
    plt.close()
    logging.info(f"Loss curve saved to {loss_plot_path}")

    #Plot and save training and validation accuracy curves
    plt.figure()
    plt.plot(epochs_list, train_accuracies, label='Training Accuracy', color='green')
    plt.plot(epochs_list, val_accuracies, label='Validation Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    acc_plot_path = os.path.join(evaluation_dir, 'accuracy_curve.png')
    plt.savefig(acc_plot_path)
    plt.close()
    logging.info(f"Accuracy curve saved to {acc_plot_path}")

    # -----------------------------------------------------------------
    # Generate and Save Classification Report
    # -----------------------------------------------------------------
    #Generate classification report
    class_report = classification_report(
        all_labels, all_preds,
        target_names=['Goodware', 'Malware'],
        zero_division=0)
    report_path = os.path.join(evaluation_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(class_report)
    logging.info(f"Classification report saved to {report_path}")

    # -----------------------------------------------------------------
    # Generate and Save Confusion Matrix
    # -----------------------------------------------------------------
    #Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Goodware', 'Malware'],
        yticklabels=['Goodware', 'Malware'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Validation Set')
    cm_plot_path = os.path.join(evaluation_dir, 'confusion_matrix.png')
    plt.savefig(cm_plot_path)
    plt.close()
    logging.info(f"Confusion matrix saved to {cm_plot_path}")

    # -----------------------------------------------------------------
    # Generate and Save ROC Curve
    # -----------------------------------------------------------------
    #Compute ROC curve and AUC
    probs = np.array(all_probs)
    all_labels_np = np.array(all_labels)
    #Use the probability of the positive class ('malware' is class 1)
    positive_probs = probs[:, 1]
    fpr, tpr, thresholds = roc_curve(all_labels_np, positive_probs)
    roc_auc = auc(fpr, tpr)

    #Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Validation Set')
    plt.legend(loc="lower right")
    roc_plot_path = os.path.join(evaluation_dir, 'roc_curve.png')
    plt.savefig(roc_plot_path)
    plt.close()
    logging.info(f"ROC curve saved to {roc_plot_path}")

    # -----------------------------------------------------------------
    # Generate and Save Metrics Bar Chart with Percentage Labels
    # -----------------------------------------------------------------
    #Prepare data for bar chart (excluding time metrics)
    metrics_to_plot = metrics_summary[metrics_summary['Metric'].isin(['F1-Score', 'Precision', 'Recall', 'Accuracy', 'ROC AUC'])]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_to_plot['Metric'], metrics_to_plot['Value'], 
                  yerr=None, capsize=4, color=['skyblue', 'lightgreen', 'salmon', 'plum', 'gold'])
    plt.title('Performance Metrics (Binary Classification)')
    plt.ylabel('Score')
    plt.ylim(0, 1.05)  # Set y-axis limits for better visualization

    # Add percentage labels on top of each bar with 2 decimal points
    for bar, value in zip(bars, metrics_to_plot['Value']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{value * 100:.2f}%", 
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    metrics_plot_path = os.path.join(evaluation_dir, f'{classification_type}_metrics_bar_chart.png')
    plt.savefig(metrics_plot_path)
    plt.close()
    logging.info(f"Metrics bar chart saved to {metrics_plot_path}")

    # -----------------------------------------------------------------
    # Save the Trained Model
    # -----------------------------------------------------------------
    #Save the trained model to the evaluation directory
    model_path = os.path.join(evaluation_dir, f'dynamic_model_{classification_type}.pt')
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved to {model_path}")

    # -----------------------------------------------------------------
    # Record Total Execution Time
    # -----------------------------------------------------------------
    #Record total training and classification time
    total_end_time = time.time()
    total_training_time = total_end_time - start_time
    logging.info(f"Total execution time for train_dynamic.py: {total_training_time/60:.2f} minutes.")

    logging.info("Training script for dynamic data completed successfully.")

# ---------------------------------------------------------------------
# Main Execution Block
# ---------------------------------------------------------------------
if __name__ == "__main__":
    """
    Main execution block for the training script.

    This block initiates the training process for dynamic malware detection by
    calling the train_model function with the specified classification type.
    """
    # -----------------------------------------------------------------
    # Record the overall start time
    # -----------------------------------------------------------------
    main_start_time = time.time()

    # -----------------------------------------------------------------
    # Start Training for Dynamic Data (Binary Classification)
    # -----------------------------------------------------------------
    classification_type = 'binary'  #Specify classification type ('binary' only for dynamic data)
    logging.info(f"Starting training for Dynamic - {classification_type.capitalize()} Classification...")
    train_model(classification_type=classification_type, num_epochs=30)
    logging.info(f"Finished training for Dynamic - {classification_type.capitalize()} Classification.")

    # -----------------------------------------------------------------
    # Record the overall end time and compute total duration
    # -----------------------------------------------------------------
    main_end_time = time.time()
    total_duration = main_end_time - main_start_time
    logging.info(f"Total execution time for train_dynamic.py: {total_duration/60:.2f} minutes.")
