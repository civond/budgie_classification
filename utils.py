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

            #print(preds.cpu().numpy())
            #print(labels.cpu().numpy())
            #input('pause')
            

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
    # Extract predictions and probs and accuracy
    preds_arr = []
    probs_arr = []
    labels_arr = []
    total_acc = 0

    # Extract activations
    activation = {}
    activation_arr = []

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach().cpu().numpy()
        return hook

    loop = tqdm.tqdm(loader)
    model.eval()

    with torch.no_grad():
        batch_length = len(loader)
        for batch_idx, (data, labels) in enumerate(loop):
            data = data.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            model.avgpool.register_forward_hook(get_activation(batch_idx))

            predictions = model(data)

            probs = torch.sigmoid(predictions)
            preds = (probs > 0.5).float()

            correct = (preds == labels).sum().item()
            accuracy = (correct / len(labels))
            #print(accuracy)
            total_acc += accuracy

            probs_arr.append(probs.cpu().numpy())
            preds_arr.append(preds.cpu().numpy())
            labels_arr.append(labels.cpu().numpy())
            #print(activation)
            #print(len(activation))
            #print(len(activation[batch_idx]))
            #print(activation[batch_idx].shape)

            for array in activation[batch_idx]:
                activation_arr.append(array)
                print(array)
                print(array.dtype)
                input('pause)')
            #print(activation_arr)
            #print(len(activation_arr))

            
    avg_acc = total_acc / batch_length
    print(avg_acc)

            #print(f"Labels: {labels}")
            #print(f"Preds: {predicted_class}")
    preds_arr = np.concatenate(preds_arr)
    labels_arr = np.concatenate(labels_arr)
    probs_arr = np.concatenate(probs_arr)

    return preds_arr, labels_arr, probs_arr, activation_arr