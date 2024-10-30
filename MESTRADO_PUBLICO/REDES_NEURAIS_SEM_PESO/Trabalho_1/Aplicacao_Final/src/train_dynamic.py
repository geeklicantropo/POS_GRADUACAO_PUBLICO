import os
import time  #Import time module for time logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  #Progress bar library
import logging
import matplotlib.pyplot as plt
import numpy as np
from utils import setup_logging
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_curve,
    auc,
    confusion_matrix
)
import seaborn as sns

# Import torch_dwn library for DWN components
import torch_dwn as dwn

def main():
    """
    Trains a Discriminative Weight Network (DWN) model for dynamic malware detection.
    """
    #Record the start time of the entire training process
    total_start_time = time.time()

    #Set up logging
    log_file = os.path.join('..', 'logs', 'train_dynamic.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    setup_logging(log_file)
    logging.info("Starting training script for dynamic data...")

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
    evaluation_dir = os.path.join(dynamic_dir, 'evaluation')
    os.makedirs(evaluation_dir, exist_ok=True)  #Create evaluation directory if it doesn't exist

    #Load training and validation data
    train_data_file = os.path.join(treated_data_dir, 'train_data.pt')
    val_data_file = os.path.join(treated_data_dir, 'val_data.pt')

    #Load data using torch.load
    train_data = torch.load(train_data_file)
    val_data = torch.load(val_data_file)

    class DynamicDataset(Dataset):
        """
        Custom Dataset class for dynamic malware data.
        """
        def __init__(self, data_dict):
            self.data = data_dict['data']
            self.labels = torch.tensor(data_dict['labels']).long()

        def __len__(self):
            return self.labels.size(0)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    #Create training and validation datasets
    train_dataset = DynamicDataset(train_data)
    val_dataset = DynamicDataset(val_data)

    #Create data loaders with reduced batch size and num_workers
    batch_size = 4  #Reduced batch size to alleviate memory issues
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    #Define the input size based on dataset
    input_size = train_dataset.data.shape[1]
    num_classes = 2  #Binary classification (Goodware vs Malware)

    #Define the DWN model and move it to the device
    '''
    This LUTLayer was used because, in the DWN original github (https://github.com/alanbacellar/DWN), it uses the LUTLayer as an example. So i will just gonna use it as well.
    '''
    model = nn.Sequential(
        #First LUTLayer with random mapping to reduce memory usage
        dwn.LUTLayer(input_size, 1000, n=4, mapping='random'),
        #Second LUTLayer with random mapping
        dwn.LUTLayer(1000, 500, n=4, mapping='random'),
        #GroupSum layer for classification
        dwn.GroupSum(num_classes, tau=1/0.3)
    ).to(device)

    #Define loss function (CrossEntropyLoss for classification tasks)
    criterion = nn.CrossEntropyLoss()
    #Define optimizer (Adam optimizer with learning rate 1e-2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    #Learning rate scheduler to adjust the learning rate during training
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=14)

    #Training parameters
    num_epochs = 30  #Number of epochs to train

    #Initialize lists to store metrics for plotting later
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_precisions = []
    val_precisions = []
    train_recalls = []
    val_recalls = []
    train_f1s = []
    val_f1s = []

    #Training loop
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch+1}/{num_epochs}")
        epoch_start_time = time.time()  #Record start time of the epoch

        model.train()  #Set model to training mode
        running_loss = 0.0  #Accumulate training loss
        correct = 0  #Number of correct predictions
        total = 0  #Total number of samples
        all_labels = []  #List to store true labels
        all_preds = []  #List to store predicted labels

        #Iterate over batches in the training loader
        for data, labels in tqdm(train_loader, desc='Training', unit='batch'):
            #Move data and labels to the device
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()  #Zero the parameter gradients

            outputs = model(data)  #Forward pass

            loss = criterion(outputs, labels)  #Compute loss

            loss.backward()  #Backward pass
            optimizer.step()  #Update weights

            running_loss += loss.item() * data.size(0)  #Update running loss

            #Compute probabilities and predictions
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)  #Update total samples
            correct += (predicted == labels).sum().item()  #Update correct predictions

            #Append labels and predictions for metrics
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            #Free up memory
            del data, labels, outputs, loss
            torch.cuda.empty_cache()

        #Compute average loss and accuracy for the epoch
        epoch_loss = running_loss / len(train_dataset)
        epoch_accuracy = correct / total

        #Compute precision, recall, and F1-score
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0)

        #Store metrics
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        train_precisions.append(precision)
        train_recalls.append(recall)
        train_f1s.append(f1)

        logging.info(f"Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, "
                     f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

        #Step the scheduler to adjust learning rate
        scheduler.step()

        #Validation loop
        model.eval()  #Set model to evaluation mode
        val_running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        all_probs = []

        val_start_time = time.time()  #Record start time of validation

        with torch.no_grad():
            #Iterate over batches in the validation loader
            for data, labels in tqdm(val_loader, desc='Validation', unit='batch'):
                #Move data and labels to the device
                data = data.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(data)  #Forward pass

                loss = criterion(outputs, labels)  # Compute loss

                val_running_loss += loss.item() * data.size(0)  #Update validation loss

                #Compute probabilities and predictions
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                #Append labels, predictions, and probabilities for metrics
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                #Free up memory
                del data, labels, outputs, loss
                torch.cuda.empty_cache()

        #Compute average validation loss and accuracy
        val_epoch_loss = val_running_loss / len(val_dataset)
        val_accuracy = correct / total

        #Compute precision, recall, and F1-score
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0)

        #Store metrics
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_accuracy)
        val_precisions.append(precision)
        val_recalls.append(recall)
        val_f1s.append(f1)

        logging.info(f"Validation Loss: {val_epoch_loss:.4f}, Accuracy: {val_accuracy:.4f}, "
                     f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

        #Log time taken for the validation
        val_end_time = time.time()
        val_duration = val_end_time - val_start_time
        logging.info(f"Validation completed in {val_duration:.2f} seconds.")

        #Log time taken for the epoch
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        logging.info(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds.")

    #Training complete, record total training time
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    logging.info(f"Total training time for Dynamic data: {total_training_time/60:.2f} minutes.")

    #Save the trained model to the evaluation directory
    model_path = os.path.join(evaluation_dir, 'dynamic_model.pt')
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved to {model_path}")

    #Plot and save training and validation loss curves
    epochs_list = np.arange(1, num_epochs + 1)

    #Loss curves
    plt.figure()
    plt.plot(epochs_list, train_losses, label='Training Loss')
    plt.plot(epochs_list, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    loss_plot_path = os.path.join(evaluation_dir, 'loss_curve.png')
    plt.savefig(loss_plot_path)
    plt.close()
    logging.info(f"Loss curve saved to {loss_plot_path}")

    #Accuracy curves
    plt.figure()
    plt.plot(epochs_list, train_accuracies, label='Training Accuracy')
    plt.plot(epochs_list, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    acc_plot_path = os.path.join(evaluation_dir, 'accuracy_curve.png')
    plt.savefig(acc_plot_path)
    plt.close()
    logging.info(f"Accuracy curve saved to {acc_plot_path}")

    #Precision curves
    plt.figure()
    plt.plot(epochs_list, train_precisions, label='Training Precision')
    plt.plot(epochs_list, val_precisions, label='Validation Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.title('Training and Validation Precision')
    prec_plot_path = os.path.join(evaluation_dir, 'precision_curve.png')
    plt.savefig(prec_plot_path)
    plt.close()
    logging.info(f"Precision curve saved to {prec_plot_path}")

    #Recall curves
    plt.figure()
    plt.plot(epochs_list, train_recalls, label='Training Recall')
    plt.plot(epochs_list, val_recalls, label='Validation Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.title('Training and Validation Recall')
    recall_plot_path = os.path.join(evaluation_dir, 'recall_curve.png')
    plt.savefig(recall_plot_path)
    plt.close()
    logging.info(f"Recall curve saved to {recall_plot_path}")

    #F1-score curves
    plt.figure()
    plt.plot(epochs_list, train_f1s, label='Training F1-score')
    plt.plot(epochs_list, val_f1s, label='Validation F1-score')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.legend()
    plt.title('Training and Validation F1-score')
    f1_plot_path = os.path.join(evaluation_dir, 'f1_score_curve.png')
    plt.savefig(f1_plot_path)
    plt.close()
    logging.info(f"F1-score curve saved to {f1_plot_path}")

    #ROC Curve and AUC for validation set
    probs = np.array(all_probs)
    all_labels_np = np.array(all_labels)
    #Use the probability of the positive class (assuming 'malware' is class 1)
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

    #Confusion Matrix for validation set
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure()
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

    logging.info("Training script for dynamic data completed successfully.")

if __name__ == "__main__":
    #Record the total time for the main function
    main_start_time = time.time()

    #Start training
    logging.info("Starting training for Dynamic data...")
    main()
    logging.info("Finished training for Dynamic data.")

    #Record the total time taken
    main_end_time = time.time()
    total_duration = main_end_time - main_start_time
    logging.info(f"Total execution time for train_dynamic.py: {total_duration/60:.2f} minutes.")
