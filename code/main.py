import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from model import Model
from utils import (
    CustomSoundDataset,
    get_sound_file_path_df,
    evaluate_model,
    get_metrics,
    plot_roc_curve,
    set_trainable_layers,
    tune_model,
)

if __name__ == "__main__":
    # Inputs#

    PROJECT_NAME = "Speech Understanding - Programming Assignment 3"
    for2_data_path = "./data/for-2seconds"
    custom_data_path = "./data/Dataset_Speech_Assignment"
    model_dict_file = "./models/Best_LA_model_for_DF.pth"

    to_tune = True
    tune_config = {
        "name": "finetune_model",
        "num_epochs": 5,
        "batch_size": 32,
        "learning_rate": 0.001,
        "start_trainable_layers": 5,
        "end_trainable_layers": 5,
        "subset_data": 0.5,
    }

    # Loading Original Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dict = torch.load(model_dict_file, map_location=device)
    model = Model(device=device)
    model.to(device)
    model = DataParallel(model)
    model.load_state_dict(model_dict)
    model_name = "pre-trained"

    # Load data and create data loader
    custom_df = get_sound_file_path_df(custom_data_path)
    test_custom_df = custom_df[custom_df["type"] == "test"].copy()
    test_custom_set = CustomSoundDataset(test_custom_df, seconds=6)
    test_custom_loader = DataLoader(test_custom_set, batch_size=8, shuffle=True)

    # Assuming you have your data_loader, model, and device set up already
    true_labels, predictions = evaluate_model(test_custom_loader, model, device)
    metrics = get_metrics(true_labels, predictions)
    fpr, tpr, thresholds, auc, eer, eer_threshold = metrics

    # Plot the ROC curve
    plot_roc_curve(
        fpr, tpr, auc, eer, filename="./outputs/evaluation_original_custom.png"
    )

    if to_tune:
        set_trainable_layers(model, tune_config)
        for2_df = get_sound_file_path_df(for2_data_path)
        for2_train_df = for2_df[for2_df["type"] == "train"].copy()
        # Sample the dataframe and shuffle the rows
        subset_train_df = for2_train_df.sample(
            frac=tune_config["subset_data"], random_state=42
        ).reset_index(drop=True)
        for2_train_set = CustomSoundDataset(subset_train_df, seconds=2)

        # Train the model and get the fine-tuned model back
        fine_tuned_model = tune_model(
            model, for2_train_set, tune_config, device, PROJECT_NAME
        )

        # Save the fine-tuned model
        torch.save(fine_tuned_model.state_dict(), "./models/fine_tuned_model.pth")
        print("Model saved successfully.")

    # # Example usage
    tuned_file_path = "./models/fine_tuned_model.pth"
    tuned_model_dict = torch.load(tuned_file_path, map_location=device)
    tuned_model = Model(device=device)
    tuned_model.to(device)
    tuned_model = DataParallel(tuned_model)
    tuned_model.load_state_dict(tuned_model_dict)

    # evaluating on testing set of for2 data
    for2_df = get_sound_file_path_df(for2_data_path)
    test_for2_df = for2_df[for2_df["type"] == "test"].copy()
    test_for2_set = CustomSoundDataset(test_for2_df, seconds=2)
    test_for2_loader = DataLoader(test_for2_set, batch_size=8, shuffle=True)

    # Evaluate the model and get ROC data
    actual, predicted = evaluate_model(test_for2_loader, tuned_model, device)
    metrics = get_metrics(actual, predicted)
    fpr, tpr, thresholds, auc, eer, eer_threshold = metrics

    # Plot the ROC curve
    plot_roc_curve(
        fpr,
        tpr,
        auc,
        eer,
        filename=f"./outputs/evaluation_finetune_for2_{tune_config['subset_data']}.png",
    )

    custom_df = get_sound_file_path_df(custom_data_path)
    test_custom_df = custom_df[custom_df["type"] == "test"].copy()
    test_custom_set = CustomSoundDataset(test_custom_df, seconds=2)
    test_custom_loader = DataLoader(test_custom_set, batch_size=8, shuffle=True)

    # Evaluate the model and get ROC data
    actual, predicted = evaluate_model(test_custom_loader, tuned_model, device)
    metrics = get_metrics(actual, predicted)
    fpr, tpr, thresholds, auc, eer, eer_threshold = metrics

    # Plot the ROC curve
    plot_roc_curve(
        fpr,
        tpr,
        auc,
        eer,
        filename=f"./outputs/evaluation_finetune_custom_{tune_config['subset_data']}.png",
    )
