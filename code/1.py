import torch
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DataParallel
import torch.nn as nn
from tqdm import tqdm
import numpy as np

import wandb
from model import Model
from utils import (
    CustomSoundDataset,
    get_sound_file_path_df,
    evaluate_model,
    get_metrics,
    plot_roc_curve,
)


def set_trainable_layers(model, config):
    """
    Freezes all layers except for a specified number of start and end layers defined in the config.
    Uses default values if not specified in the config.

    :param model: The model whose layers are to be partially frozen.
    :param config: Configuration dictionary which may contain 'start_trainable_layers' and 'end_trainable_layers'.
    """
    start_layers = config.get(
        "start_trainable_layers", 5
    )  # Default to 5 if not specified
    end_layers = config.get("end_trainable_layers", 5)  # Default to 5 if not specified

    children = list(model.named_parameters())
    num_children = len(children)
    for i, (_, param) in enumerate(children):
        if i < start_layers or i >= num_children - end_layers:
            param.requires_grad = True
        else:
            param.requires_grad = False


def train_model(model, dataset, config, device, project):
    # Initialize Weights & Biases
    wandb.init(project=project, config=config)

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Setup DataLoaders for training and validation
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # Define loss function with default weights
    weights = torch.tensor(config.get("weights", [0.1, 0.9]), dtype=torch.float32).to(
        device
    )
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Define optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
    )

    # Training and validation loop
    for epoch in range(config["num_epochs"]):
        model.train()
        total_train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * inputs.size(0)
        train_loss = total_train_loss / len(train_dataset)

        model.eval()
        total_val_loss = 0
        val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
        with torch.no_grad():
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item() * inputs.size(0)
        val_loss = total_val_loss / len(val_dataset)

        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    wandb.finish()

    return model


#####################################################################################################


to_train = False
# PROJECT_NAME = "Speech Understanding - Programming Assignment 3"
PROJECT_NAME = "TEST"

# Configurations
test_original_config = {
    "name": "infer_original_model",
    "audio_length": 6,
    "batch_size": 8,
}

tune_config = {
    "name": "finetune_model",
    "num_epochs": 5,
    "batch_size": 32,
    "learning_rate": 0.001,
    "start_trainable_layers": 5,
    "end_trainable_layers": 5,
    "subset_data": 0.05,
}

test_tuned_config = {
    "name": "infer_tuned_model",
    "audio_length": 6,
    "batch_size": 8,
}


# Loading Original Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file_path = "./models/Best_LA_model_for_DF.pth"
model_dict = torch.load(file_path, map_location=device)
model = Model(device=device)
model.to(device)
model = DataParallel(model)
model.load_state_dict(model_dict)
model_name = "pre-trained"


# Load data and create data loader
custom_data_path = "./data/Dataset_Speech_Assignment"
custom_df = get_sound_file_path_df(custom_data_path)
test_custom_df = custom_df[custom_df["type"] == "test"].copy()
test_custom_set = CustomSoundDataset(
    test_custom_df, seconds=test_original_config["audio_length"]
)
test_custom_loader = DataLoader(
    test_custom_set, batch_size=test_original_config["batch_size"], shuffle=True
)

# Assuming you have your data_loader, model, and device set up already
true_labels, predictions = evaluate_model(test_custom_loader, model, device)
metrics = get_metrics(true_labels, predictions)
fpr, tpr, thresholds, auc, eer, eer_threshold = metrics

# Plot the ROC curve
plot_roc_curve(fpr, tpr, auc, eer, filename="./outputs/evaluation_original_custom.png")


if to_train:
    set_trainable_layers(model, tune_config)
    for2_data_path = "./data/for-2seconds"
    for2_df = get_sound_file_path_df(for2_data_path)
    for2_train_df = for2_df[for2_df["type"] == "train"].copy()
    # Sample the dataframe and shuffle the rows
    subset_train_df = for2_train_df.sample(
        frac=tune_config["subset_data"], random_state=42
    ).reset_index(drop=True)
    for2_train_set = CustomSoundDataset(subset_train_df, seconds=2)

    # Train the model and get the fine-tuned model back
    fine_tuned_model = train_model(
        model, for2_train_set, tune_config, device, PROJECT_NAME
    )

    # Save the fine-tuned model
    torch.save(fine_tuned_model.state_dict(), "./models/fine_tuned_model2.pth")
    print("Model saved successfully.")


# # Example usage
tuned_file_path = "./models/fine_tuned_model.pth"
tuned_model_dict = torch.load(tuned_file_path, map_location=device)
tuned_model = Model(device=device)
tuned_model.to(device)
tuned_model = DataParallel(tuned_model)
tuned_model.load_state_dict(tuned_model_dict)

# evaluating on testing set of for2 data
for2_data_path = "./data/for-2seconds"
for2_df = get_sound_file_path_df(for2_data_path)
test_for2_df = for2_df[for2_df["type"] == "test"].copy()
test_for2_set = CustomSoundDataset(test_for2_df, seconds=2)
test_for2_loader = DataLoader(test_for2_set, batch_size=8, shuffle=True)

# Evaluate the model and get ROC data
actual, predicted = evaluate_model(test_for2_loader, tuned_model, device)
metrics = get_metrics(actual, predicted)
fpr, tpr, thresholds, auc, eer, eer_threshold = metrics

# Plot the ROC curve
plot_roc_curve(fpr, tpr, auc, eer, filename="./outputs/evaluation_finetune_for2.png")


# custom_data_path = "./data/Dataset_Speech_Assignment"
# custom_df = get_sound_file_path_df(custom_data_path)
# test_custom_df = custom_df[custom_df["type"] == "test"].copy()
# test_custom_set = CustomSoundDataset(test_custom_df, seconds=2)
# test_custom_loader = DataLoader(test_custom_set, batch_size=8, shuffle=True)

# Evaluate the model and get ROC data
actual, predicted = evaluate_model(test_custom_loader, tuned_model, device)
metrics = get_metrics(actual, predicted)
fpr, tpr, thresholds, auc, eer, eer_threshold = metrics

# Plot the ROC curve
plot_roc_curve(fpr, tpr, auc, eer, filename="./outputs/evaluation_finetune_custom.png")


# # # Initialize wandb
# # # wandb.init(project="your_project_name", entity="your_username")
# # # # Assuming you have your data_loader, model, and device set up already
# # # true_labels, predictions = evaluate_model(data_loader, model, device, wandb_log=True)
# # # metrics = get_metrics(true_labels, predictions, wandb_log=True)
# # # fpr, tpr, thresholds, auc, eer, eer_threshold = metrics
# # # # Log ROC curve
# # # roc_data = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}
# # # wandb.log(
# # #     {
# # #         "ROC Curve": wandb.plot.line_series(
# # #             roc_data,
# # #             title="ROC Curve",
# # #             xaxis="False Positive Rate",
# # #             yaxis="True Positive Rate",
# # #         )
# # #     }
# # # )
