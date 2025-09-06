

import os
import time
import torch
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score
from tempfile import TemporaryDirectory
from torchvision import transforms

from utils.CNN_utils import DualViewDataset, train_model
from models.models import MultiViewResNet50_adaptive
import numpy as np
import json
import hashlib
from itertools import product
import hashlib
import json

EXPERIMENTS = [
]
class Repetitions:
            done_exps = set()
    

def get_transform(mode):
    if mode == "RGB":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

def get_all_transforms(modes):
    return [get_transform(m) for m in modes]

def run_experiment(config, device):
    print(f"\n>> Ejecutando experimento: {config['exp_name']}")
    os.makedirs("models", exist_ok=True)
    os.makedirs("gates", exist_ok=True)

    transforms_list = get_all_transforms(config["modes"])

    dataset = DualViewDataset(
        roots=config["views"],
        modes=config["modes"],
        transform=transforms_list,
        conjunct_transform=None,
        datasetType=config["datasetType"],
        model_name=config['exp_name']  # ← se guarda como RGB_plus_VDVI.pt
    )

    train_dataset, val_dataset = dataset.split(train_ratio=0.85)

    if config['use_sampler']:
        labels = [s[-1] for s in train_dataset.samples]
        counts = torch.bincount(torch.tensor(labels))
        weights_per_class = 1. / counts.float()
        sample_weights = weights_per_class[torch.tensor(labels)]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=8, sampler=sampler),
            'val': DataLoader(val_dataset, batch_size=8, shuffle=False)
        }
    else:
        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=8, shuffle=True),
            'val': DataLoader(val_dataset, batch_size=8, shuffle=False)
        }

    dataset_sizes = {k: len(dataloaders[k].dataset) for k in dataloaders}
    channels = config["channels"] if config["datasetType"] == "multiview" else [sum(config["channels"])]

    model = MultiViewResNet50_adaptive(
        channels=channels,
        resnet_types=config["backbones"],
        datasetType=config["datasetType"],
        num_classes=len(train_dataset.classes),
        gated=config["gated"],
        pretrained=True
    ).to(device)


    criterion = torch.nn.CrossEntropyLoss()



    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    try:
        model, train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s, val_gates = train_model(
            model=model,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=10,
            model_name=config['exp_name']
        )

        model_file = f"models/{config['exp_name']}.pt"
        torch.save(model.state_dict(), model_file)

        gates_file = "None"
        if val_gates is not None:
            gates_file = f"gates/gates_{config['exp_name']}.pt"
            torch.save(val_gates, gates_file)

        return {
            "name": config["exp_name"],
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "final_train_acc": train_accs[-1],
            "final_val_acc": val_accs[-1],
            "final_train_f1": train_f1s[-1],
            "final_val_f1": val_f1s[-1],
            "gates_file": gates_file,
            "model_file": model_file,
        }

    except Exception as e:
        print(f"❌ Error en experimento {config['exp_name']}: {e}")
        return None



def main():
    reps = Repetitions(EXPERIMENTS)
    while reps.next() is not None:
        reps.print(current=True)
        reps.realize()


            

        #model_path = f"trained_models/{config['exp_name']}.pt"
        #if os.path.exists(model_path) or config["exp_name"] in done_exps:
         #   print(f"✔️  Saltando {config['exp_name']} (ya realizado)")
          #  continue

        #result = run_experiment(config, device)
        #if result is not None:
         #   df_results = pd.concat([df_results, pd.DataFrame([result])], ignore_index=True)
          #  df_results.to_csv(results_path, index=False)

if __name__ == "__main__":
    main()