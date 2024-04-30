from model import Model
import torch
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from utils import (
    CustomSoundDataset,
    get_sound_file_path_df,
    evaluate_model,
    get_metrics,
    plot_roc_curve,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(device=device)
model.to(device)

file_path = "./models/Best_LA_model_for_DF.pth"
dictModel = torch.load(file_path, map_location=device)
model = DataParallel(model)
model.load_state_dict(dictModel)
model_name = "pre-trained"


# Draw inference on custom_data

#Load data and create data loader
data_path = "./data/Dataset_Speech_Assignment"
df = get_sound_file_path_df(data_path)
dataset = CustomSoundDataset(df, seconds=4)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Evaluate the model and get ROC data
actual, predicted = evaluate_model(data_loader, model, device)
metrics = get_metrics(actual, predicted)
fpr, tpr, thresholds, auc, eer, eer_threshold = metrics

# Plot the ROC curve
plot_roc_curve(fpr, tpr, auc, eer)

# # eval_data_dir = 'datasets/Dataset_Speech_Assignment'
# # fine_tune_data_dir = 'datasets/for-2seconds/training'
# for name, param in model.named_parameters():
#     # if "classifier" not in name:
#     #     param.requires_grad = False
#     print(name, param.requires_grad, param.shape)
