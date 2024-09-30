import os
import warnings
from tqdm import tqdm 
from model import build_transformer
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from config import get_config, get_weights_file_path, latest_weights_file_path

from dataloader import TimeSeriesDataset
from datapreprocess import load_data, load_data_local, load_single_with_indicators

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

from matplotlib import pyplot as plt

def get_ds(config, strategy=None):
    # Load and split the data
    #file_path = './dataset/AAPL_data_5min_1.csv'
    #train_data, val_data = load_data(file_path)
    
    file_path = '../data/strategy_data.csv'
    if strategy:
        train_data, val_data = load_single_with_indicators (file_path, strategy)
        print(f'Loading data for strategy {strategy}')
    else:
        train_data, val_data = load_data(file_path)

    # print("before",len(train_data))

    # Create instances of the TimeSeriesDataset
    train_dataset = TimeSeriesDataset(train_data)
    val_dataset = TimeSeriesDataset(val_data)

    # print("after",len(train_dataset))
    # print(type(train_dataset))
    # for idx in range(2):
    #     sample = train_dataset[idx]  # Access each sample in the subset
    #     # Process or print the sample data as needed
    #     print(sample)

    batch_size = config['batch_size']  # Define your batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    # # Example usage of DataLoader
    # for batch in train_loader:
    #     inputs = batch['encoder_input']  # Input sequences
    #     targets = batch['label']  # Target values

    #     # Your training loop goes here using inputs and targets
    #     # Replace this example with your model training code
    #     print("Input shape:", inputs.shape)
    #     print("Target shape:", targets)
    #     break  # Break after the first batch for demonstration

    return train_loader,val_loader

def get_model(config, vocab_tgt_len):
    model = build_transformer(vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model

def train_model(config, target_len = 1):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    # if (device == 'cuda'):
    #     print(f"Device name: {torch.cuda.get_device_name(device.index)}")
    #     print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    # elif (device == 'mps'):
    #     print(f"Device name: <mps>")
    # else:
    #     print("NOTE: If you have a GPU, consider using it for training.")
    #     print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
    #     print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader = get_ds(config)

    model = get_model(config, target_len).to(device)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.MSELoss().to(device)
    
    
    MSE_track = []
    MAE_track = []
    
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = None
            decoder_mask = None
            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)  ## why not seq_len-1?
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)

            #The ProjectionLayer class below is responsible for converting the output of the model into a probability distribution over the vocabulary, where we select each output token from a vocabulary of possible tokens.
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # parameter may change, need to process this in model.py
            # (B, vocab_size)
            # scaled by first observation
            prediction = torch.mean(encoder_input.view(10,7), dim=0) + torch.matmul(torch.std(
                encoder_input.view(10,7), dim=0) / torch.sqrt(torch.tensor(10)), proj_output[:,-1].view(7))

            # Compare the output with the label
            label = batch['label'].to(device) # (B, vocab_size)

            # Compute the loss using a simple cross entropy
            #loss = loss_fn(proj_output.view(-1, 1), label.view(-1))
            loss = loss_fn(label.view(-1), prediction)
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        MAE, MSE = run_validation(model, val_dataloader, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
        MAE_track.append(MAE)
        MSE_track.append(MSE)
        
        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
        
    
def train_val_single_indicator(config, target_len = 1, strategy=None):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    # if (device == 'cuda'):
    #     print(f"Device name: {torch.cuda.get_device_name(device.index)}")
    #     print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    # elif (device == 'mps'):
    #     print(f"Device name: <mps>")
    # else:
    #     print("NOTE: If you have a GPU, consider using it for training.")
    #     print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
    #     print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader = get_ds(config, strategy)
    
    print(train_dataloader)

    model = get_model(config, target_len).to(device)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.MSELoss().to(device)
    
    
    MSE_track = []
    MAE_track = []
    
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = None
            decoder_mask = None
            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)  ## why not seq_len-1?
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)

            #The ProjectionLayer class below is responsible for converting the output of the model into a probability distribution over the vocabulary, where we select each output token from a vocabulary of possible tokens.
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)
            '''
            print(torch.mean(encoder_input.view(10,7), dim=0).shape)
            print((torch.std(
                encoder_input.view(10,7), dim=0) / torch.sqrt(torch.tensor(10))).shape)
            print(proj_output[:,-1].repeat(1, 7).reshape(-1).shape)
            '''
            # parameter may change, need to process this in model.py
            # (B, vocab_size)
            # scaled by first observation
            prediction = torch.mean(encoder_input.view(10,7), dim=0) + torch.matmul(torch.std(
                encoder_input.view(10,7), dim=0) / torch.sqrt(torch.tensor(10)), proj_output[:,-1].repeat(1, 7).reshape(-1))

            # Compare the output with the label
            label = batch['label'].to(device) # (B, vocab_size)

            # Compute the loss using a simple cross entropy
            #loss = loss_fn(proj_output.view(-1, 1), label.view(-1))
            loss = loss_fn(label.view(-1), prediction)
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        MAE, MSE = run_validation(model, val_dataloader, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
        MAE_track.append(MAE)
        MSE_track.append(MSE)
        
        #plt.plot(MAE_track)
        #plt.pause(0.05)
        
        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
        
    #plt.plot(MAE_track)
    #plt.plot(MSE_track)
    #plt.show()
    
    min_weight = MAE_track.index(min(MAE_track))
    
    initial_epoch = 0
    global_step = 0
    preload = str(min_weight)
    if len(preload) == 1: preload = '0'+preload
    model_filename = get_weights_file_path(config, preload)

    print(f'Validating model using {model_filename}')
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])
    initial_epoch = state['epoch'] + 1
    optimizer.load_state_dict(state['optimizer_state_dict'])
    global_step = state['global_step']
    
    source_texts = []
    expected = []
    predicted = []

    model.eval()
    
    with torch.no_grad():
        for batch in val_dataloader:
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            #encoder_mask = batch["decoder_mask"].to(device) # (b, 1, 1, seq_len)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = None
            decoder_mask = None

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # parameter may change, need to process this in model.py
            # (B, vocab_size)
            prediction = (torch.mean(encoder_input.view(10, 7), dim=0) + torch.matmul(
                torch.std(encoder_input.view(10, 7), dim=0) / torch.sqrt(torch.tensor(10)), proj_output[:, -1].repeat(1, 7).reshape(-1)))[0]

            # Compare the output with the label
            label = batch['label'].to(device)[0][0] # (B, seq_len)
            #print(proj_output, label)

            source_texts.append(encoder_input[0][-7].item())
            expected.append(label.item())
            predicted.append(prediction.item())
    
    plt.figure()
    plt.plot(expected, label='expected')
    plt.plot(predicted, label='predicted')
    plt.title(f'strategy_{strategy}')
    plt.legend()
    plt.savefig(f'strategy_{strategy}.png')
    
    df_result = pd.DataFrame({'Previous Day Price': source_texts, 'Expected': expected, 'Prediction': predicted})
    df_result['direction'] = 1 * ((df_result['Expected'] - df_result['Previous Day Price']) * (df_result['Prediction'] - df_result['Previous Day Price']) > 0)
    direction_accuracy = df_result['direction'].mean()

    df_metric = pd.DataFrame({'Direction Accuracy': [direction_accuracy], 
                              'MAE': [MAE_track[min_weight]],
                              'MSE': [MSE_track[min_weight]],
                              'weight': [min_weight]})
    
    # create a excel writer object
    with pd.ExcelWriter(f"strategy_{strategy}.xlsx") as writer:
       
        # use to_excel function and specify the sheet_name and index 
        # to store the dataframe in specified sheet
        df_result.to_excel(writer, sheet_name="result", index=False)
        df_metric.to_excel(writer, sheet_name="metric", index=False)
    


def run_validation(model, validation_ds, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    tensor_mae, tensor_mse = torch.tensor(()), torch.tensor(())
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            #encoder_mask = batch["decoder_mask"].to(device) # (b, 1, 1, seq_len)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = None
            decoder_mask = None

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # parameter may change, need to process this in model.py
            # (B, vocab_size)
            prediction = torch.mean(encoder_input.view(10, 7), dim=0) + torch.matmul(
                torch.std(encoder_input.view(10, 7), dim=0) / torch.sqrt(torch.tensor(10)), proj_output[:, -1].repeat(1, 7).reshape(-1))

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)
            #print(proj_output, label)

            ae = torch.abs(torch.sub(prediction, label))
            se = torch.square(ae)
            tensor_mae = torch.cat((tensor_mae, ae),dim=0)
            tensor_mse = torch.cat((tensor_mse, se),dim=0)
    MAE = torch.mean(tensor_mae, dim=0)[0]
    MSE = torch.mean(tensor_mse, dim=0)[0]
    #print("validation MAE: ", torch.mean(tensor_mae, dim=0))
    #print("validation MSE: ", torch.mean(tensor_mse, dim=0))
    return MAE, MSE


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    
    iNumOfAssets = 7 # this is the single strat data + 6 indicators
    
    for i in range(1, 35):
        print(f'Running strategy {i}')
        train_val_single_indicator(config, target_len = 1, strategy=i)
    
    #train_model(config, iNumOfAssets)
    #validation_single(config, iNumOfAssets)
