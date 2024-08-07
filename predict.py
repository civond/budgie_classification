import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torchvision.transforms import v2
import torchvision.models as models
import pandas as pd
from datetime import datetime
import pandas as pd
import json

from utils import *
from model import *

# Hyperparameters
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_EPOCHS = 15
NUM_WORKERS = 8
PIN_MEMORY = True

LOAD_MODEL = True
TRAIN = False
CHECKPOINT_PATH = "checkpoints/2024_07_30_04_53_54.pth.tar"
dataset_pth = "data/testing_set.csv"

# Generate predictions
def main():
    # Load the model for binary classiication
    num_classes = 1
    model = models.efficientnet_b0()
    model.classifier[1] = nn.Linear(1280, num_classes)

    #model = ResNet152(img_channel=3, num_classes=num_classes)
    model = model.to(DEVICE)

    # Import dataset
    df = pd.read_csv(dataset_pth)
    df.replace({'label': {1: 0, 2: 1}}, inplace=True) # Reformat labels

    # Define validation / test transformation
    test_df = df[df['fold'] == 5]
    #test_df = df[df['fold'].isin([0,1,2,3,4,5])]
    test_df.reset_index(drop=True, inplace=True)
    
    test_transform = v2.Compose([
            v2.ToTensor(),
            v2.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std= np.sqrt([1.0, 1.0, 1.0]) # variance is std**2
                )
        ])

    # If load checkpoint == True
    if LOAD_MODEL: 
        print("Loading model.")
        load_checkpoint(torch.load(CHECKPOINT_PATH), model)

    
    if TRAIN == False and LOAD_MODEL == True:
        # Get the time
        now = datetime.now()
        timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
        temp_preds_path = f"csv/preds_{timestamp}.csv"
        df_list = []

        # Predict
        train = False
        shuffle=False
        test_loader = get_loader(
                test_df, BATCH_SIZE, test_transform,
                NUM_WORKERS, train, shuffle, PIN_MEMORY
            )
        
        [preds, labels, probs, activations] = predict(test_loader, 
                        model, 
                        device=DEVICE)
        
        
        # This will put probabilities into the thing. 
        # Use a sigmoid function to turn to classification

        test_df = test_df.copy()

        test_df.loc[:, 'probs'] = probs
        test_df.loc[:, 'activations'] = activations

        
        #test_df.loc[:,'preds'] = preds

 
        # Concat Lists
        print(f"\tWriting {temp_preds_path}")
        columns_to_keep = ['onset', 'offset','label', 'preds', 'probs', 'bird', 'labels2', 'activations']
        test_df = test_df.drop(test_df.columns.difference(columns_to_keep), axis=1)
        test_df.to_csv(temp_preds_path, sep=',', index=False)

        # Save DataFrame to a JSON file
        columns_to_keep = ['label', 'probs', 'activations']
        test_df = test_df.drop(test_df.columns.difference(columns_to_keep), axis=1)
        for index, row in test_df.iterrows():
            temp_path = f"json/{index}.json"
            row_dict = row.to_dict()
            row_dict['activations'] = row_dict['activations'].tolist()
            with open(temp_path, 'w') as json_file:
                json.dump(row_dict, json_file, indent=4)

if __name__ == "__main__":
    main()