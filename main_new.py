import os
import torch
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm

from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# Import your custom modules
from encoder import Encoder
from decoder import Decoder
from model import ED
from net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_encoder_params, convgru_decoder_params
from data.nc import NcDataset
from earlystopping import EarlyStopping
from hybrid_loss import HybridLoss

def main(args):
    """
    Main function, containing the complete training and validation pipeline.
    """
    # 1. Set up the device (key change: adaptive device detection)
    # Automatically detect CUDA, Apple Silicon (MPS), or CPU
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        print(f"Using CUDA device: {device}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS backend.")
    else:
        device = torch.device("cpu")
        print("CUDA not found. Using CPU.")

    # 2. Set random seed for reproducibility
    random_seed = 2025
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        # The following two settings balance performance and reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 3. Prepare DataLoaders
    print("Loading data...")
    train_folder = NcDataset(is_train=True,
                           root='data/',
                           n_frames_input=args.frames_input,
                           n_frames_output=args.frames_output)
    valid_folder = NcDataset(is_train=False,
                           root='data/',
                           n_frames_input=args.frames_input,
                           n_frames_output=args.frames_output)

    # Best practice: shuffle the training data (shuffle=True)
    train_loader = DataLoader(train_folder,
                              batch_size=args.batch_size,
                              shuffle=True, 
                              num_workers=args.num_workers)
    valid_loader = DataLoader(valid_folder,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers)
    print("Data loaded.")

    # 4. Model selection and initialization
    if args.convlstm:
        encoder_params = convlstm_encoder_params
        decoder_params = convlstm_decoder_params
        model_type = 'convlstm'
    else:  # Use ConvGRU by default
        encoder_params = convgru_encoder_params
        decoder_params = convgru_decoder_params
        model_type = 'convgru'

    encoder = Encoder(encoder_params[0], encoder_params[1])
    decoder = Decoder(decoder_params[0], decoder_params[1])
    net = ED(encoder, decoder)
    
    # Key change: send the model to the determined device
    net.to(device)

    # Use DataParallel for multi-GPU training (only active with multiple GPUs)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        net = nn.DataParallel(net)

    # 5. Set up paths and logging
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir = os.path.join('./save_model', f'{timestamp}_{model_type}')
    run_dir = os.path.join('./runs', f'{timestamp}_{model_type}')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    tb = SummaryWriter(run_dir)

    # 6. Initialize loss function, optimizer, learning rate scheduler, and early stopping
    loss_function = HybridLoss(lambda_grad=args.lambda_grad).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4, verbose=True)
    early_stopping = EarlyStopping(patience=20, verbose=True, path=os.path.join(save_dir, 'checkpoint.pth'))
    
    cur_epoch = 0
    # Optional: Load an existing model to resume training (spelling mistake fixed)
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pth.tar')
    if os.path.exists(checkpoint_path):
        print(f'==> Resuming from checkpoint: {checkpoint_path}')
        model_info = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(model_info['state_dict'])
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch'] + 1

    # 7. Training and validation loop
    for epoch in range(cur_epoch, args.epochs):
        # --- Training ---
        net.train()
        train_losses = []
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for _, inputs, label in train_pbar:
            # Key change: send data to the device
            inputs = inputs.to(device)  # B,S,C,H,W
            label = label.to(device)   # B,S,C,H,W

            optimizer.zero_grad()
            pred = net(inputs)
            loss = loss_function(pred, label)
            
            # Correction: loss.item() is already the averaged value, no need to divide by batch_size again
            train_losses.append(loss.item())
            
            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0)
            optimizer.step()

            train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_train_loss = np.mean(train_losses)
        tb.add_scalar('Loss/Train', avg_train_loss, epoch)

        # --- Validation ---
        net.eval()
        valid_losses = []
        valid_pbar = tqdm(valid_loader, desc=f"Epoch {epoch}/{args.epochs} [Valid]")
        with torch.no_grad():
            for _, inputs, label in valid_pbar:
                inputs = inputs.to(device)
                label = label.to(device)
                
                pred = net(inputs)
                loss = loss_function(pred, label)
                valid_losses.append(loss.item())
                valid_pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        avg_valid_loss = np.mean(valid_losses)
        tb.add_scalar('Loss/Valid', avg_valid_loss, epoch)
        tb.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch {epoch}: Avg Train Loss: {avg_train_loss:.6f}, Avg Valid Loss: {avg_valid_loss:.6f}")

        # Update learning rate
        pla_lr_scheduler.step(avg_valid_loss)

        # Early stopping check
        model_dict = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        early_stopping(avg_valid_loss, model_dict)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
            
    tb.close()
    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spatio-Temporal Prediction Model Training")
    
    # Model and training parameters
    parser.add_argument('--convlstm', action='store_true', help='Use ConvLSTM as the base cell.')
    parser.add_argument('--convgru', action='store_true', help='Use ConvGRU as the base cell (default).')
    parser.add_argument('--batch_size', default=5, type=int, help='Mini-batch size.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to train.')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate.')
    parser.add_argument('--lambda_grad', default=0.2, type=float, help='Weight for the gradient loss term in HybridLoss.')
    
    # Data parameters
    parser.add_argument('--frames_input', default=6, type=int, help='Number of input frames.')
    parser.add_argument('--frames_output', default=6, type=int, help='Number of output frames to predict.')
    
    # System parameters
    parser.add_argument('--num_workers', default=4, type=int, help='Number of worker threads for DataLoader.')
    parser.add_argument('--gpu_id', default=0, type=int, help='ID of the GPU to use if CUDA is available.')

    args = parser.parse_args()

    # If neither is specified, default to using ConvGRU
    if not args.convlstm and not args.convgru:
        args.convgru = True
        print("No model type specified, defaulting to ConvGRU.")

    print("Running with the following configuration:")
    print(args)
    
    main(args)
