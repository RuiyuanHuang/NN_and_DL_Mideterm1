import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import time
import os
from torch.utils.tensorboard import SummaryWriter # Import SummaryWriter

# --- Configuration ---
DATA_DIR = 'caltech-101/101_ObjectCategories/101_ObjectCategories'
NUM_CLASSES = 101
BATCH_SIZE = 32 # Adjusted for CPU and memory
IMAGE_SIZE = (128, 128) # Smaller image size for faster training on CPU
NUM_EPOCHS = 5 # Limited epochs to meet the 10-minute constraint
LEARNING_RATE_FC = 0.001 # Learning rate for the new fully connected layer
LEARNING_RATE_FINETUNE = 0.0001 # Smaller learning rate for pre-trained layers
TRAIN_SPLIT_RATIO = 0.8
RANDOM_SEED = 0

# --- 1. Data Loading and Preprocessing ---
def get_data_loaders(data_dir, image_size, batch_size, train_split_ratio, random_seed): # filter_list removed
    """
    Loads the Caltech-101 dataset and creates training and validation DataLoaders.
    """
    print(f"Loading data from: {data_dir}")

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet stats
    ])

    try:
        # Using ImageFolder directly
        full_dataset = ImageFolder(root=data_dir, transform=transform)
        print(f"Dataset loaded. Total classes found: {len(full_dataset.classes)}")
        if len(full_dataset.classes) == 0:
            print(f"Error: No classes found. Check if '{data_dir}' contains subdirectories for each category.")
            return None, None, 0
        if len(full_dataset) == 0:
            print(f"Error: Dataset is empty. Check image files in {data_dir}")
            return None, None, len(full_dataset.classes)

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"Please ensure the Caltech-101 dataset is correctly placed in the '{data_dir}' directory.")
        print("And that it's not empty or corrupted, and unwanted class folders (like 'BACKGROUND_Google') are removed if necessary.")
        return None, None, 0

    # Splitting dataset
    num_train = int(len(full_dataset) * train_split_ratio)
    num_val = len(full_dataset) - num_train

    if num_train == 0 or num_val == 0:
        print(f"Error: Not enough data to split into training and validation sets. Num train: {num_train}, Num val: {num_val}")
        return None, None, len(full_dataset.classes)

    train_dataset, val_dataset = random_split(full_dataset, [num_train, num_val], generator=torch.Generator().manual_seed(random_seed))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of classes: {len(full_dataset.classes)}")

    return train_loader, val_loader, len(full_dataset.classes)


# --- 2. Model Definition ---
def get_model(num_classes, learning_rate_fc, learning_rate_finetune):
    """
    Loads a pre-trained ResNet-18 model and modifies its classifier.
    Sets different learning rates for the new classifier and the pre-trained layers.
    """
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Modify the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes) # New layer has requires_grad=True by default

    # Set up optimizer with different learning rates
    # Parameters of the newly initialized fully connected layer
    nn.init.xavier_uniform_(model.fc.weight);
    fc_params = model.fc.parameters()

    # Parameters of the pre-trained layers (we will unfreeze some of them for fine-tuning)
    # For ResNet, let's fine-tune layer4 and upwards (more specific features)
    finetune_params = []
    for name, param in model.named_parameters():
        if "fc" not in name: # Exclude the already handled fc layer
             # Unfreeze later layers for fine-tuning
            if 'layer4' in name or 'layer3' in name: # Example: fine-tune layer3 and layer4
                param.requires_grad = True
                finetune_params.append(param)
            else:
                param.requires_grad = False # Keep earlier layers frozen


    optimizer = optim.Adam([
        {'params': fc_params, 'lr': learning_rate_fc},
        {'params': finetune_params, 'lr': learning_rate_finetune}
    ], lr=learning_rate_fc) # Default lr, though overridden by group specific lrs

    return model, optimizer

# --- 3. Training and Evaluation ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, writer): # Add writer
    """
    Trains the model and evaluates it on the validation set.
    Logs training and validation metrics to TensorBoard.
    """
    print(f"Training on {device}...")
    global_step = 0 # For logging batch-wise training loss
    for epoch in range(num_epochs):
        start_time_epoch = time.time()
        model.train()
        running_loss_train = 0.0 # Renamed for clarity
        correct_train = 0
        total_train = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss_train += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Log training loss per batch to TensorBoard
            writer.add_scalar('Loss/train_batch', loss.item(), global_step)
            global_step += 1

            if (i + 1) % 20 == 0: # Print every 20 batches
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        epoch_loss_train = running_loss_train / len(train_loader.dataset)
        epoch_acc_train = correct_train / total_train
        
        # Log average training loss and accuracy for the epoch
        writer.add_scalar('Loss/train_epoch', epoch_loss_train, epoch)
        writer.add_scalar('Accuracy/train_epoch', epoch_acc_train, epoch)

        # Validation
        model.eval()
        running_loss_val = 0.0 # Renamed for clarity
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss_val += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_loss_val = running_loss_val / len(val_loader.dataset)
        epoch_acc_val = correct_val / total_val

        # Log validation loss and accuracy for the epoch to TensorBoard
        writer.add_scalar('Loss/validation_epoch', epoch_loss_val, epoch)
        writer.add_scalar('Accuracy/validation_epoch', epoch_acc_val, epoch)

        end_time_epoch = time.time()
        epoch_duration = end_time_epoch - start_time_epoch

        print(f"Epoch {epoch+1}/{num_epochs} took {epoch_duration:.2f}s")
        print(f"  Train Loss: {epoch_loss_train:.4f}, Train Acc: {epoch_acc_train:.4f}")
        print(f"  Val Loss: {epoch_loss_val:.4f}, Val Acc: {epoch_acc_val:.4f}")

    print("Finished Training")
    return model


# --- 4. Main Execution ---
if __name__ == '__main__':
    print("Starting Caltech-101 classification task...")
    overall_start_time = time.time()

    # Initialize SummaryWriter
    # Creates a 'runs' directory with a sub-directory for this specific run
    run_name = f"Caltech101_ResNet18_{time.strftime('%Y%m%d-%H%M%S')}"
    log_dir = os.path.join('runs', run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if str(device) == "cuda":
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")


    # Get DataLoaders
    train_loader, val_loader, num_actual_classes = get_data_loaders(
        data_dir=DATA_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        train_split_ratio=TRAIN_SPLIT_RATIO,
        # filter_list=FILTER_LIST, # Removed
        random_seed=RANDOM_SEED
    )

    if train_loader is None or val_loader is None:
        print("Failed to load data. Exiting.")
        writer.close() # Close writer if data loading fails
        exit()

    if num_actual_classes != NUM_CLASSES and num_actual_classes > 0 :
        print(f"Warning: Number of classes detected ({num_actual_classes}) differs from NUM_CLASSES ({NUM_CLASSES}).")
        print(f"Using {num_actual_classes} as the number of classes.")
        current_num_classes = num_actual_classes
    elif num_actual_classes == 0:
        print("Error: No classes were loaded. Exiting.")
        writer.close() # Close writer if no classes loaded
        exit()
    else:
        current_num_classes = NUM_CLASSES


    # Get Model and Optimizer
    model, optimizer = get_model(
        num_classes=current_num_classes,
        learning_rate_fc=LEARNING_RATE_FC,
        learning_rate_finetune=LEARNING_RATE_FINETUNE
    )
    model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Train the model
    print(f"Starting training for {NUM_EPOCHS} epoch(s)...")
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device, writer)

    # Close the SummaryWriter
    writer.close()

    # Save the trained model (optional)
    # model_save_path = "caltech101_resnet18_finetuned.pth"
    # torch.save(trained_model.state_dict(), model_save_path)
    # print(f"Trained model saved to {model_save_path}")

    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time
    print(f"Total execution time: {total_duration:.2f} seconds.")
    if total_duration > 600: # 10 minutes
        print("Warning: Training took longer than 10 minutes.")

    print("Task completed.") 