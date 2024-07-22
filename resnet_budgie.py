import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torchvision.transforms import v2
import pandas as pd
from datetime import datetime
import pandas as pd

from utils import *
from model import *

# Hyperparameters
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_EPOCHS = 15
NUM_WORKERS = 8
PIN_MEMORY = True

LOAD_MODEL = False
TRAIN = True
dataset_pth = "data/meta.csv"

def main():
    # Load the model for binary classiication
    num_classes = 1
    model = ResNet152(img_channel=3, num_classes=num_classes) 
    model = model.to(DEVICE)

    # If not doing binary classification, use cross entropy loss
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    # Define train transformation
    train_transform = v2.Compose([
        v2.ToTensor(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std= np.sqrt([1.0, 1.0, 1.0]) # variance is std**2
        )
        ])
    
    # Define validation / test transformation
    test_transform = v2.Compose([
            v2.ToTensor(),
            v2.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std= np.sqrt([1.0, 1.0, 1.0]) # variance is std**2
                )
        ])
    
    # Import dataset
    df = pd.read_csv(dataset_pth)
    df.replace({'label': {1: 0, 2: 1}}, inplace=True) # Reformat labels

    # Define training Set
    train_df = df[df['fold'].isin([0,1,2,3])]
    train_df.reset_index(drop=True, inplace=True)
    train = True
    shuffle=False

    # Get train loader
    train_loader = get_loader(
            train_df, BATCH_SIZE, train_transform,
            NUM_WORKERS, train, shuffle, PIN_MEMORY
        )
    
    # Define validation Set
    valid_df = df[df['fold'] == 4]
    valid_df.reset_index(drop=True, inplace=True)
    train = False

    # Get validation loader
    valid_loader = get_loader(
            valid_df, BATCH_SIZE, test_transform,
            NUM_WORKERS, train, shuffle, PIN_MEMORY
        )
    
    # If load checkpoint == True
    if LOAD_MODEL: 
        print("Loading model.")
        load_checkpoint(torch.load("checkpoints/2024_07_03_16_25_45.pth.tar"), model)

    ### Train ###
    if TRAIN == True:
        # Track metrics
        train_acc_running = []; train_loss_running = []
        val_acc_running = []; val_loss_running = []
        
        # Loop through epochs
        for epoch in range(NUM_EPOCHS):
            # Train function
            print(f"\nEpoch: {epoch}")
            [train_acc, train_loss] = train_fn(
                train_loader, model, 
                optimizer, loss_fn, scaler)

            # Validation function
            [val_acc, val_loss] = validate(
                valid_loader, model,
                loss_fn, device=DEVICE
                )

            # Append metrics
            train_acc_running.append(train_acc)
            train_loss_running.append(train_loss)
            val_acc_running.append(val_acc)
            val_loss_running.append(val_loss)

        # Get current timestamp
        now = datetime.now()
        timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")

        # Save model and metrics
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(state = checkpoint, filename= f"checkpoints/{timestamp}.pth.tar")
        save_metrics(timestamp, train_loss_running, train_acc_running, val_loss_running, val_acc_running)

    ### Predictions ###
    if TRAIN == False and LOAD_MODEL == True:
        # Get the time
        now = datetime.now()
        timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
        temp_preds_path = f"csv/preds_{timestamp}.csv"
        df_list = []

        # Chunk the testing set
        i = 0
        chunk_size = 10_000  # Adjust based on memory capacity
        print(f"Chunk Size: {chunk_size}")

        for chunk in pd.read_csv("meta_twobird.csv", chunksize=chunk_size):
            print(f"Loading Chunk {i}")
            print(f"Len: {len(chunk)}")
            #print(chunk)

            chunk = chunk.reset_index(drop=True)

            # Predict
            train = False
            shuffle=False
            test_loader = get_loader(
                    chunk,
                    BATCH_SIZE,
                    test_transform,
                    NUM_WORKERS,
                    train,
                    shuffle,
                    PIN_MEMORY
                )
            
            print(f"Generating predictions for chunk {i}")
            preds, labels = predict(test_loader, 
                            model, 
                            device=DEVICE)
            
            

            chunk['preds'] = preds
            
            df_list.append(chunk)
            i+=1
        
        # Concat Lists
        print(f"\tWriting {temp_preds_path}")
        merged_df = pd.concat(df_list, axis=0)
        columns_to_keep = ['onset', 'offset','label', 'preds', 'bird']
        merged_df = merged_df.drop(merged_df.columns.difference(columns_to_keep), axis=1)
        merged_df.to_csv(temp_preds_path, sep=',', index=False)

if __name__ == "__main__":
    main()