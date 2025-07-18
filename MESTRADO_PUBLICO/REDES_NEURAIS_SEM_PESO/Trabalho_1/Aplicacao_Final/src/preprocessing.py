import os
import time  #Import time module for time logging
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm  #For progress bars
import torch
import torch_dwn as dwn  #import torch_dwn
import h5py  #For efficient data storage
import gc    #For garbage collection
import logging

from utils import setup_logging  #Import the logging setup function


class StaticPreprocessor:
    """
    A class to handle preprocessing of static data (images) for DWN.

    This class reads image data, fits a Distributive Thermometer encoding,
    and preprocesses images by binarizing them. The processed data is saved
    directly to disk using HDF5 to minimize memory usage.
    """

    def __init__(self, data_dir):
        """
        Initializes the StaticPreprocessor with the given data directory.

        Parameters:
            data_dir (str): The directory where the image data is located.
        """
        self.data_dir = data_dir
        self.thermometer = None  #Will be initialized during fitting
        self.class_to_idx = None  #Will store the class to index mapping

    def read_image_paths(self):
        """
        Reads image paths from the given directory and returns a list of image paths and their labels.

        Returns:
            image_paths (list of str): List of image file paths.
            labels (list of str): List of labels corresponding to each image.
        """
        logging.info(f"Reading image paths from {self.data_dir}...")
        classes = sorted(os.listdir(self.data_dir))
        image_paths = []
        labels = []
        logging.info(f"Found {len(classes)} classes in {self.data_dir}: {classes}")
        for class_name in tqdm(classes, desc='Reading classes', unit='class'):
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                logging.warning(f"Skipping {class_name} as it is not a directory.")
                continue
            image_files = sorted(os.listdir(class_dir))
            for image_file in image_files:
                image_path = os.path.join(class_dir, image_file)
                image_paths.append(image_path)
                labels.append(class_name)
        logging.info(f"Total images found: {len(image_paths)}")
        return image_paths, labels

    def preprocess_images(self, output_file, fit_thermometer=False, num_bits=3, classification_type='binary'):
        """
        Preprocesses images by reading them iteratively, fitting the thermometer (if needed), and binarizing.
        Saves the binarized data directly to disk using HDF5 to minimize memory usage.

        Parameters:
            output_file (str): File path to save the preprocessed data (HDF5 file).
            fit_thermometer (bool): Whether to fit the thermometer (should be True for training data).
            num_bits (int): Number of bits per feature in the encoding.
            classification_type (str): 'binary' or 'multiclass' classification.
        """
        #Record the start time of the preprocessing
        start_time = time.time()

        image_paths, labels = self.read_image_paths()
        num_samples = len(image_paths)
        logging.info("Processing images iteratively and saving to disk...")

        if classification_type == 'binary':
            #Map class names to indices for binary classification
            class_to_idx = {'goodware': 0, 'malware': 1}
            self.class_to_idx = class_to_idx  # Save for later use

            #Update labels for binary classification
            labels_binary = ['goodware' if label == 'Other' else 'malware' for label in labels]
            labels_to_use = labels_binary
        elif classification_type == 'multiclass':
            #Map class names to indices for multi-class classification
            class_names = sorted(list(set(labels)))
            class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
            self.class_to_idx = class_to_idx  #Save for later use
            labels_to_use = labels
        else:
            raise ValueError("Invalid classification_type. Choose 'binary' or 'multiclass'.")

        #Convert labels to indices
        labels_indices = [class_to_idx[label] for label in labels_to_use]

        #Determine the size of the binarized data
        #Process the first image to get the shape
        sample_image = Image.open(image_paths[0]).convert('L')
        sample_tensor = torch.from_numpy(np.array(sample_image)).float().view(-1)
        if fit_thermometer and self.thermometer is None:
            logging.info("Fitting Distributive Thermometer on first image...")
            #Initialize and fit the Distributive Thermometer
            self.thermometer = dwn.DistributiveThermometer(num_bits=num_bits, feature_wise=True)
            self.thermometer.fit(sample_tensor.unsqueeze(0))
            logging.info("Thermometer encoding fitted.")
        elif self.thermometer is None:
            raise ValueError("Thermometer not fitted. Please fit on training data first.")
        binarized_sample = self.thermometer.binarize(sample_tensor.unsqueeze(0))
        binarized_sample_flat = binarized_sample.reshape(-1)
        binarized_shape = binarized_sample_flat.shape

        #Delete sample variables to free memory
        del sample_image, sample_tensor, binarized_sample, binarized_sample_flat
        gc.collect()

        #Create HDF5 file to store data
        with h5py.File(output_file, 'w') as h5f:
            data_dataset = h5f.create_dataset('data', shape=(num_samples, binarized_shape[0]), dtype='uint8')
            label_dataset = h5f.create_dataset('labels', shape=(num_samples,), dtype='int')

            #Process images iteratively
            for idx in tqdm(range(num_samples), desc='Processing images', unit='image'):
                image_path = image_paths[idx]
                label_idx = labels_indices[idx]
                try:
                    #Open the image and convert it to grayscale
                    img = Image.open(image_path).convert('L')
                    #Convert the image to a torch tensor
                    img_tensor = torch.from_numpy(np.array(img)).float()
                    #Flatten the image
                    img_tensor_flat = img_tensor.view(-1)
                    #Binarize the image data
                    binarized_img = self.thermometer.binarize(img_tensor_flat.unsqueeze(0))
                    #Flatten the binarized data
                    binarized_img_flat = binarized_img.reshape(-1)
                    #Convert to numpy array and ensure data type is uint8
                    binarized_img_flat_np = binarized_img_flat.numpy().astype('uint8')
                    #Write data and label to HDF5 dataset
                    data_dataset[idx] = binarized_img_flat_np
                    label_dataset[idx] = label_idx

                    #Explicitly delete variables to free memory
                    del img, img_tensor, img_tensor_flat, binarized_img, binarized_img_flat, binarized_img_flat_np
                    gc.collect()

                except Exception as e:
                    logging.error(f"Error processing image {image_path}: {e}")

        #Record the end time and calculate duration
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"Preprocessing images completed in {duration/60:.2f} minutes.")
        logging.info("All images processed and saved to HDF5 file.")

    def save_class_mapping(self, class_to_idx, filename):
        """
        Saves the class to index mapping to a file.

        Parameters:
            class_to_idx (dict): Dictionary mapping class names to indices.
            filename (str): Filename to save the mapping.
        """
        logging.info(f"Saving class mapping to {filename}...")
        torch.save(class_to_idx, filename)
        logging.info(f"Class mapping saved to {filename}.")


class DynamicPreprocessor:
    """
    A class to handle preprocessing of dynamic data (CSV file with features) for DWN.

    This class reads the CSV data, balances the dataset, splits it into training and validation sets,
    fits a Distributive Thermometer encoding on the training data, and binarizes the data.
    """

    def __init__(self, csv_file_path):
        """
        Initializes the DynamicPreprocessor with the given CSV file path.

        Parameters:
            csv_file_path (str): The path to the CSV file containing the data.
        """
        self.csv_file_path = csv_file_path
        self.df = None
        self.thermometer = None  #Will be initialized during fitting

    def read_csv(self):
        """
        Reads the CSV file into a pandas DataFrame.
        """
        logging.info(f"Reading CSV data from {self.csv_file_path}...")
        self.df = pd.read_csv(self.csv_file_path)
        logging.info(f"CSV data loaded with shape {self.df.shape}")

    def fit_thermometer(self, X, num_bits=3):
        """
        Fits the Distributive Thermometer encoder on the training features.

        Parameters:
            X (torch.Tensor): Tensor of training features.
            num_bits (int): Number of bits per feature in the encoding.
        """
        logging.info("Fitting Distributive Thermometer on training data...")
        #Initialize and fit the Distributive Thermometer
        self.thermometer = dwn.DistributiveThermometer(num_bits=num_bits, feature_wise=True)
        self.thermometer.fit(X)
        logging.info("Thermometer encoding fitted.")

    def binarize_data(self, X):
        """
        Binarizes data using the fitted Distributive Thermometer encoding.

        Parameters:
            X (torch.Tensor): Tensor of features.

        Returns:
            binarized_data (torch.Tensor): Tensor of binarized data.
        """
        logging.info("Binarizing data using Distributive Thermometer encoding...")
        #Binarize using the fitted thermometer
        binarized_data = self.thermometer.binarize(X)
        #Flatten the binarized data (keeping batch dimension)
        binarized_data_flat = binarized_data.flatten(start_dim=1)
        logging.info("Binarization complete.")
        return binarized_data_flat

    def preprocess_data(self, num_bits=3):
        """
        Performs preprocessing on the data, including handling missing values, balancing, splitting, and binarization.

        Parameters:
            num_bits (int): Number of bits per feature in the encoding.

        Returns:
            X_train_binarized (torch.Tensor): Binarized training features.
            X_val_binarized (torch.Tensor): Binarized validation features.
            y_train (list): Training labels.
            y_val (list): Validation labels.
        """
        #Record the start time of the preprocessing
        start_time = time.time()
        logging.info("Starting preprocessing of dynamic data...")

        #Handle missing values by filling them with 0
        logging.info("Handling missing values by filling with 0...")
        self.df.fillna(0, inplace=True)

        #Update the label column name to 'malware' or 'label' based on CSV file
        label_column_name = 'malware'  #Update if CSV has a different label column

        #Check if the label column exists
        if label_column_name not in self.df.columns:
            logging.error(f"Error: '{label_column_name}' column not found in CSV data.")
            return None, None, None, None

        #Display counts per class before balancing
        counts = self.df[label_column_name].value_counts()
        logging.info(f"Counts per class before balancing: {counts.to_dict()}")

        #Balance the dataset by sampling an equal number from each class
        logging.info("Balancing the dataset by sampling equal number of samples from each class...")
        min_count = counts.min()
        df_balanced = self.df.groupby(label_column_name, group_keys=False).apply(
            lambda x: x.sample(n=min_count, random_state=42)
        ).reset_index(drop=True)
        logging.info(f"Balanced data shape: {df_balanced.shape}")

        #Separate features and labels
        logging.info("Separating features and labels...")
        X = df_balanced.drop([label_column_name, 'hash'], axis=1, errors='ignore')  #Drop 'hash' if exists
        y = df_balanced[label_column_name]

        #Convert to torch tensors
        logging.info("Converting data to torch tensors...")
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_list = y.tolist()

        #Split data into training and validation sets (70% train, 30% validation)
        logging.info("Splitting data into training and validation sets (70% train, 30% validation)...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_tensor, y_list, test_size=0.3, random_state=42, stratify=y_list
        )

        logging.info(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")

        #Fit the thermometer on training data
        self.fit_thermometer(X_train, num_bits=num_bits)

        #Binarize training and validation data
        X_train_binarized = self.binarize_data(X_train)
        X_val_binarized = self.binarize_data(X_val)

        #Free up memory
        del X_tensor, X_train, X_val
        gc.collect()

        #Record the end time and calculate duration
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"Dynamic data preprocessing completed in {duration/60:.2f} minutes.")

        return X_train_binarized, X_val_binarized, y_train, y_val

    def save_preprocessed_data(self, X_binarized, y, filename):
        """
        Saves the preprocessed data and labels to a file using torch.save.

        Parameters:
            X_binarized (torch.Tensor): Tensor of binarized data.
            y (list): List of labels.
            filename (str): Filename to save the data.
        """
        logging.info(f"Saving preprocessed data to {filename}...")
        torch.save({'data': X_binarized, 'labels': y}, filename)
        logging.info(f"Data saved to {filename}.")


def main():
    """
    Main function to orchestrate the preprocessing of static and dynamic data.

    This function sets up logging, initializes preprocessors for static and dynamic data,
    and processes the data accordingly.
    """
    #Set up logging
    log_file = os.path.join('..', 'logs', 'preprocessing.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    setup_logging(log_file)
    logging.info("Starting preprocessing script...")

    #Record the start time of the entire preprocessing
    total_start_time = time.time()

    #Get the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logging.info(f"Script directory: {script_dir}")

    #Move up one directory to the project root
    project_dir = os.path.dirname(script_dir)
    logging.info(f"Project directory: {project_dir}")

    # ========================
    # Static Data Preprocessing
    # ========================

    logging.info("Starting static data preprocessing...")

    #Paths for static data
    static_dir = os.path.join(project_dir, 'dados', 'static')
    train_data_dir = os.path.join(static_dir, 'train')
    val_data_dir = os.path.join(static_dir, 'val')
    processed_data_dir = os.path.join(static_dir, 'processed')
    treated_data_dir = os.path.join(static_dir, 'treated_data')
    binary_treated_data_dir = os.path.join(treated_data_dir, 'binary')
    multiclass_treated_data_dir = os.path.join(treated_data_dir, 'multiclass')

    #Create the processed data directories if they don't exist
    os.makedirs(processed_data_dir, exist_ok=True)
    os.makedirs(treated_data_dir, exist_ok=True)
    os.makedirs(binary_treated_data_dir, exist_ok=True)
    os.makedirs(multiclass_treated_data_dir, exist_ok=True)

    # ---------------------------
    # Binary Classification
    # ---------------------------

    #Record the start time of binary classification preprocessing
    binary_start_time = time.time()

    #Preprocess training images for binary classification
    logging.info("Preprocessing training images for binary classification...")
    static_preprocessor_train = StaticPreprocessor(train_data_dir)
    train_output_file_binary = os.path.join(binary_treated_data_dir, 'train_data.h5')
    static_preprocessor_train.preprocess_images(
        output_file=train_output_file_binary,
        fit_thermometer=True,
        num_bits=3,
        classification_type='binary'
    )

    #Save the fitted thermometer for later use
    thermometer_path_binary = os.path.join(processed_data_dir, 'thermometer_binary.pt')
    logging.info(f"Saving fitted thermometer to {thermometer_path_binary}...")
    torch.save(static_preprocessor_train.thermometer, thermometer_path_binary)
    logging.info(f"Thermometer saved to {thermometer_path_binary}")

    #Save class mapping
    class_mapping_path_binary = os.path.join(processed_data_dir, 'class_mapping_binary.pt')
    static_preprocessor_train.save_class_mapping(static_preprocessor_train.class_to_idx, class_mapping_path_binary)

    #Preprocess validation images for binary classification
    logging.info("Preprocessing validation images for binary classification...")
    #Initialize the preprocessor with the same thermometer and class mapping
    static_preprocessor_val = StaticPreprocessor(val_data_dir)
    static_preprocessor_val.thermometer = static_preprocessor_train.thermometer  #Use the same thermometer
    static_preprocessor_val.class_to_idx = static_preprocessor_train.class_to_idx  #Use same class mapping
    val_output_file_binary = os.path.join(binary_treated_data_dir, 'val_data.h5')
    static_preprocessor_val.preprocess_images(
        output_file=val_output_file_binary,
        fit_thermometer=False,
        classification_type='binary'
    )

    #Record the end time and calculate duration for binary preprocessing
    binary_end_time = time.time()
    binary_duration = binary_end_time - binary_start_time
    logging.info(f"Static data preprocessing for binary classification completed in {binary_duration/60:.2f} minutes.")

    # ---------------------------
    # Multi-Class Classification
    # ---------------------------

    #Record the start time of multi-class classification preprocessing
    multiclass_start_time = time.time()

    #Preprocess training images for multi-class classification
    logging.info("Preprocessing training images for multi-class classification...")
    static_preprocessor_train_multi = StaticPreprocessor(train_data_dir)
    train_output_file_multi = os.path.join(multiclass_treated_data_dir, 'train_data.h5')
    static_preprocessor_train_multi.preprocess_images(
        output_file=train_output_file_multi,
        fit_thermometer=True,
        num_bits=3,
        classification_type='multiclass'
    )

    #Save the fitted thermometer for later use
    thermometer_path_multi = os.path.join(processed_data_dir, 'thermometer_multiclass.pt')
    logging.info(f"Saving fitted thermometer to {thermometer_path_multi}...")
    torch.save(static_preprocessor_train_multi.thermometer, thermometer_path_multi)
    logging.info(f"Thermometer saved to {thermometer_path_multi}")

    #Save class mapping
    class_mapping_path_multi = os.path.join(processed_data_dir, 'class_mapping_multiclass.pt')
    static_preprocessor_train_multi.save_class_mapping(static_preprocessor_train_multi.class_to_idx, class_mapping_path_multi)

    #Preprocess validation images for multi-class classification
    logging.info("Preprocessing validation images for multi-class classification...")
    #Initialize the preprocessor with the same thermometer and class mapping
    static_preprocessor_val_multi = StaticPreprocessor(val_data_dir)
    static_preprocessor_val_multi.thermometer = static_preprocessor_train_multi.thermometer  #Use the same thermometer
    static_preprocessor_val_multi.class_to_idx = static_preprocessor_train_multi.class_to_idx  #Use same class mapping
    val_output_file_multi = os.path.join(multiclass_treated_data_dir, 'val_data.h5')
    static_preprocessor_val_multi.preprocess_images(
        output_file=val_output_file_multi,
        fit_thermometer=False,
        classification_type='multiclass'
    )

    #Record the end time and calculate duration for multi-class preprocessing
    multiclass_end_time = time.time()
    multiclass_duration = multiclass_end_time - multiclass_start_time
    logging.info(f"Static data preprocessing for multi-class classification completed in {multiclass_duration/60:.2f} minutes.")

    # =========================
    # Dynamic Data Preprocessing
    # =========================

    logging.info("Starting dynamic data preprocessing...")

    #Record the start time of dynamic data preprocessing
    dynamic_start_time = time.time()

    #Path to the dynamic CSV file using project directory
    dynamic_dir = os.path.join(project_dir, 'dados', 'dynamic')
    dynamic_csv_path = os.path.join(dynamic_dir, 'top_1000_pe_imports.csv')

    #Check if the CSV file exists
    if not os.path.isfile(dynamic_csv_path):
        logging.error(f"Dynamic CSV file not found at {dynamic_csv_path}.")
        return
    else:
        logging.info(f"Found dynamic CSV file at {dynamic_csv_path}.")

    #Paths for dynamic data using project directory
    dynamic_processed_data_dir = os.path.join(dynamic_dir, 'processed')
    dynamic_treated_data_dir = os.path.join(dynamic_dir, 'treated_data')

    #Create the processed data directories if they don't exist
    os.makedirs(dynamic_processed_data_dir, exist_ok=True)
    os.makedirs(dynamic_treated_data_dir, exist_ok=True)

    #Initialize the DynamicPreprocessor
    dynamic_preprocessor = DynamicPreprocessor(dynamic_csv_path)

    #Read the CSV data
    dynamic_preprocessor.read_csv()

    #Preprocess the data
    X_train_binarized, X_val_binarized, y_train, y_val = dynamic_preprocessor.preprocess_data(num_bits=3)

    #Check if preprocessing was successful
    if X_train_binarized is not None:
        #Save the preprocessed training data
        train_filename = os.path.join(dynamic_treated_data_dir, 'train_data.pt')
        dynamic_preprocessor.save_preprocessed_data(X_train_binarized, y_train, train_filename)

        #Save the preprocessed validation data
        val_filename = os.path.join(dynamic_treated_data_dir, 'val_data.pt')
        dynamic_preprocessor.save_preprocessed_data(X_val_binarized, y_val, val_filename)

        #Save the fitted thermometer for later use
        thermometer_path = os.path.join(dynamic_processed_data_dir, 'thermometer.pt')
        logging.info(f"Saving fitted thermometer to {thermometer_path}...")
        torch.save(dynamic_preprocessor.thermometer, thermometer_path)
        logging.info(f"Thermometer saved to {thermometer_path}")
    else:
        logging.error("Dynamic data preprocessing failed due to errors in the data.")

    #Record the end time and calculate duration for dynamic data preprocessing
    dynamic_end_time = time.time()
    dynamic_duration = dynamic_end_time - dynamic_start_time
    logging.info(f"Dynamic data preprocessing completed in {dynamic_duration/60:.2f} minutes.")

    #Total preprocessing time
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    logging.info(f"Total preprocessing time: {total_duration/60:.2f} minutes.")

    logging.info("Preprocessing script completed successfully.")


if __name__ == "__main__":
    main()
