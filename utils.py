from dataset import *
from torch.utils.data import DataLoader, ConcatDataset
import tqdm
import torch
import pandas as pd

def save_checkpoint(state, filename = "my_checkpoint.pth.tar"):
    print("\t-> Saving checkpoint.")
    torch.save(state, filename)

def save_metrics(timestamp, train_loss_running, train_acc_running, val_loss_running, val_acc_running):
    temp_data = {
            "train_loss" : train_loss_running,
            "train_acc" : train_acc_running, 
            "val_loss" : val_loss_running,
            "val_acc" : val_acc_running
            }
    temp_df = pd.DataFrame(temp_data)
    temp_csv_path = f"csv/stats_{timestamp}.csv"
    print(f"\tWriting {temp_csv_path}")
    temp_df.to_csv(temp_csv_path, sep=',', index=False)

def load_checkpoint(checkpoint, model):
    print("-> Loading checkpoint.")
    model.load_state_dict(checkpoint["state_dict"])
    
def get_loader(dataframe,
               batch_size,
               transform,
               num_workers=12,
               train=True,
               shuffle=False,
               pin_memory=True):
    
    ds = ImageDataset(
        dataframe, 
        train=train,
        transform=transform
        )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return loader

def train_fn(loader, model, optimizer, loss_fn, scaler, device='cuda'):
    loop = tqdm.tqdm(loader)
    total_acc = 0
    total_loss = 0
    num_batches = len(loader)
    
    for batch_idx, (data, labels) in enumerate(loop):
        data = data.to(device)
        labels = labels.float().unsqueeze(1).to(device) # Get rid of float, unsqueeze if not using BCE.
        #print(f"Len Labels: {len(labels)}")

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, labels)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Get loss
        total_loss += loss.item()

        # Pass through sigmoid function to calculate accuracy
        probs = torch.sigmoid(predictions)
        preds = (probs > 0.5).float()
        correct = (preds == labels).sum().item()

        
        accuracy = correct / len(labels)
        #print(accuracy)
        total_acc += accuracy
        

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        
    avg_acc = total_acc / num_batches
    avg_loss = total_loss / num_batches
    print(f"Train Avg_Acc: {avg_acc}, Avg_Loss: {avg_loss}")
    return avg_acc, avg_loss

def validate(loader, model, loss_fn, device="cuda"):
    loop = tqdm.tqdm(loader)
    total_loss = 0
    total_acc = 0
    model.eval()
    
    with torch.no_grad():
        batch_length = len(loader)


        for batch_idx, (data, labels) in enumerate(loop):
            data = data.to(device)
            labels = labels.float().unsqueeze(1).to(device) # Get rid of float, unsqueeze if not using BCE.

            predictions = model(data)
            loss = loss_fn(predictions, labels)
            total_loss += loss

            # Pass through sigmoid function
            probs = torch.sigmoid(predictions)
            preds = (probs > 0.5).float()

            correct = (preds == labels).sum().item()
            accuracy = (correct / len(labels))
            #print(accuracy)
            total_acc += accuracy

            # update tqdm loop
            loop.set_postfix(loss=loss.item())
        
    avg_acc = total_acc / batch_length
    avg_loss = total_loss / batch_length # Compute the average loss across batches
    
    print(f"Validation Avg_Acc: {avg_acc}, Avg_Loss: {avg_loss}")
    return avg_acc, avg_loss.item()

def predict(loader, model, device="cuda"):
    preds_arr = []
    labels_arr = []
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            #labels = labels.to(device)

            outputs = model(images)
            _, predicted_class = torch.max(outputs, 1)
            preds_arr.append(predicted_class.cpu().numpy())
            labels_arr.append(labels)

            #print(f"Labels: {labels}")
            #print(f"Preds: {predicted_class}")
    preds_arr = np.concatenate(preds_arr)
    labels_arr = np.concatenate(labels_arr)
    #print(preds_arr)
    #print(len(preds_arr))
    return preds_arr, labels_arr