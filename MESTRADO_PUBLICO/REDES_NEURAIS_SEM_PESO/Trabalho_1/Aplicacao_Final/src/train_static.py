import os
import time  #Import time module for time logging for reporting later
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

#Import torch_dwn library for DWN components
import torch_dwn as dwn

def train_model(classification_type='binary'):
    """
    Trains a Discriminative Weight Network (DWN) model for static malware detection.

    Parameters:
        classification_type (str): 'binary' for binary classification, 'multiclass' for multi-class classification.
    """
    #Record the start time of the training process
    start_time = time.time()

    #Set up logging
    log_file = os.path.join('..', 'logs', f'train_static_{classification_type}.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    setup_logging(log_file)
    logging.info(f"Starting training script for static data ({classification_type} classification)...")

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

    #Define paths for static data
    static_dir = os.path.join(project_dir, 'dados', 'static')
    treated_data_dir = os.path.join(static_dir, 'treated_data', classification_type)
    evaluation_dir = os.path.join(static_dir, 'evaluation', classification_type)
    os.makedirs(evaluation_dir, exist_ok=True)  #Create evaluation directory if it doesn't exist

    #Load class mapping (dictionary mapping class names to indices)
    class_mapping_path = os.path.join(static_dir, 'processed', f'class_mapping_{classification_type}.pt')
    class_to_idx = torch.load(class_mapping_path)
    idx_to_class = {v: k for k, v in class_to_idx.items()}  #Reverse mapping
    num_classes = len(class_to_idx)  #Number of classes

    #Create datasets using custom HDF5Dataset class
    train_data_file = os.path.join(treated_data_dir, 'train_data.h5')
    val_data_file = os.path.join(treated_data_dir, 'val_data.h5')
    train_dataset = HDF5Dataset(train_data_file)
    val_dataset = HDF5Dataset(val_data_file)

    #Create data loaders with reduced batch size and num_workers
    batch_size = 4  #Reduced batch size to alleviate memory issues. It takes longer but a bigger batch_size was not working
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    #Define the input size based on dataset
    input_size = train_dataset.data.shape[1]

    if classification_type == 'binary':
        #For binary classification, set appropriate number of classes
        num_output_classes = 2
        group_sum_classes = 2
    else:
        #For multi-class classification
        num_output_classes = num_classes
        group_sum_classes = num_classes

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
        dwn.GroupSum(group_sum_classes, tau=1/0.3)
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

        # Iterate over batches in the training loader
        for data, labels in tqdm(train_loader, desc='Training', unit='batch'):
            # Move data and labels to the device
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

            #Append labels and predictions for metrics
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            #Free up memory
            del data, labels, outputs, loss
            torch.cuda.empty_cache()

        #Compute average loss and accuracy for the epoch
        epoch_loss = running_loss / len(train_dataset)
        epoch_accuracy = correct / total

        train_losses.append(epoch_loss)  #Store training loss
        train_accuracies.append(epoch_accuracy)  #Store training accuracy

        logging.info(f"Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

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

        val_losses.append(val_epoch_loss)  #Store validation loss
        val_accuracies.append(val_accuracy)  #Store validation accuracy

        logging.info(f"Validation Loss: {val_epoch_loss:.4f}, Accuracy: {val_accuracy:.4f}")

        #Log time taken for the epoch
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        logging.info(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds.")

    #Training complete, record total training time
    end_time = time.time()
    total_training_time = end_time - start_time
    logging.info(f"Total training time for {classification_type} classification: {total_training_time/60:.2f} minutes.")

    #Save the trained model to the evaluation directory
    model_path = os.path.join(evaluation_dir, f'static_model_{classification_type}.pt')
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
    plt.title(f'Training and Validation Loss ({classification_type.capitalize()} Classification)')
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
    plt.title(f'Training and Validation Accuracy ({classification_type.capitalize()} Classification)')
    acc_plot_path = os.path.join(evaluation_dir, 'accuracy_curve.png')
    plt.savefig(acc_plot_path)
    plt.close()
    logging.info(f"Accuracy curve saved to {acc_plot_path}")

    #Generate classification report
    class_report = classification_report(
        all_labels, all_preds,
        target_names=[idx_to_class[i] for i in range(num_classes)],
        zero_division=0)
    report_path = os.path.join(evaluation_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(class_report)
    logging.info(f"Classification report saved to {report_path}")

    #Generate and save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=[idx_to_class[i] for i in range(num_classes)],
        yticklabels=[idx_to_class[i] for i in range(num_classes)])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Validation Set ({classification_type.capitalize()} Classification)')
    cm_plot_path = os.path.join(evaluation_dir, 'confusion_matrix.png')
    plt.savefig(cm_plot_path)
    plt.close()
    logging.info(f"Confusion matrix saved to {cm_plot_path}")

    #For binary classification, generate ROC curve and AUC
    if classification_type == 'binary':
        #Compute ROC AUC
        probs = np.array(all_probs)
        all_labels_np = np.array(all_labels)
        #Use the probability of the positive class (e.g., 'malware' as class 1)
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
    #Record the total time for training both models
    total_start_time = time.time()

    #Train for binary classification
    logging.info("Starting training for Static - Binary Classification...")
    binary_start_time = time.time()
    train_model(classification_type='binary')
    binary_end_time = time.time()
    logging.info("Finished training for Static - Binary Classification.")
    binary_training_time = binary_end_time - binary_start_time
    logging.info(f"Total time for Static - Binary Classification training: {binary_training_time/60:.2f} minutes.")

    #Train for multi-class classification
    logging.info("Starting training for Static - Multi-class Classification...")
    multiclass_start_time = time.time()
    train_model(classification_type='multiclass')
    multiclass_end_time = time.time()
    logging.info("Finished training for Static - Multi-class Classification.")
    multiclass_training_time = multiclass_end_time - multiclass_start_time
    logging.info(f"Total time for Static - Multi-class Classification training: {multiclass_training_time/60:.2f} minutes.")

    #Log total time taken for both trainings
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    logging.info(f"Total training time for Static data: {total_training_time/60:.2f} minutes.")
