#!/usr/bin/env python3
import time
import torch
import os
import itertools
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from config.train_options import TrainOptions
from data.datasets import UnalignedStyleDataset
from models.adain_cycle_gan import AdaINStyleCycleGAN
from utils.image_pool import ImagePool


def main():
    """Main training function.
    
    This function parses command-line options, creates the model, dataset,
    and optimizer, and trains the model.
    """
    # Parse command-line options
    opt = TrainOptions().parse()
    
    # Set the random seed for reproducibility
    torch.manual_seed(opt.seed if hasattr(opt, 'seed') else 42)
    
    # Create the dataset and dataloader
    dataset = UnalignedStyleDataset(opt)
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,
        num_workers=opt.num_threads
    )
    
    # Create directories for checkpoints and samples
    os.makedirs(os.path.join(opt.checkpoints_dir, opt.name), exist_ok=True)
    
    # Create the model
    model = AdaINStyleCycleGAN(opt)
    
    # Set up the image buffers for discriminator training
    fake_A_pool = ImagePool(opt.pool_size)
    fake_B_pool = ImagePool(opt.pool_size)
    
    # Set up the optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(model.netG_A.parameters(), model.netG_B.parameters()),
        lr=opt.lr,
        betas=(opt.beta1, 0.999)
    )
    
    optimizer_D = torch.optim.Adam(
        itertools.chain(model.netD_A.parameters(), model.netD_B.parameters()),
        lr=opt.lr,
        betas=(opt.beta1, 0.999)
    )
    
    # Assign optimizers to the model
    model.optimizer_G = optimizer_G
    model.optimizer_D = optimizer_D
    
    # Set up the learning rate schedulers
    def lambda_rule(epoch):
        """Calculate the decay factor for learning rate."""
        lr_decay_factor = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
        return lr_decay_factor
    
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda_rule)
    
    # Load checkpoint if needed
    if opt.continue_train:
        checkpoint_path = os.path.join(opt.checkpoints_dir, opt.name, 'latest_net.pth')
        if os.path.exists(checkpoint_path):
            print(f'Loading the model from {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path)
            model.netG_A.load_state_dict(checkpoint['netG_A'])
            model.netG_B.load_state_dict(checkpoint['netG_B'])
            model.netD_A.load_state_dict(checkpoint['netD_A'])
            model.netD_B.load_state_dict(checkpoint['netD_B'])
            if 'optimizer_G' in checkpoint:
                optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            if 'optimizer_D' in checkpoint:
                optimizer_D.load_state_dict(checkpoint['optimizer_D'])
            if 'epoch' in checkpoint:
                opt.epoch_count = checkpoint['epoch'] + 1
    
    # Set up TensorBoard writer
    writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, 'logs'))
    
    # Training loop
    total_iterations = 0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        
        # Iterate over the dataset
        for i, data in enumerate(dataloader):
            iter_start_time = time.time()
            
            # Increment iteration counters
            total_iterations += opt.batch_size
            epoch_iter += opt.batch_size
            
            # Set the model input data
            model.set_input(data)
            
            # Perform one step of optimization
            model.optimize_parameters()
            
            # Display progress and results
            if total_iterations % opt.print_freq == 0:
                losses = {
                    'G_A': model.loss_G_A.item(),
                    'G_B': model.loss_G_B.item(),
                    'cycle_A': model.loss_cycle_A.item(),
                    'cycle_B': model.loss_cycle_B.item(),
                    'D_A': model.loss_D_A.item(),
                    'D_B': model.loss_D_B.item()
                }
                if hasattr(model, 'loss_style') and model.loss_style != 0:
                    losses['style'] = model.loss_style.item()
                
                # Print current losses
                t_data = iter_start_time - iter_data_time
                t_comp = time.time() - iter_start_time
                print(f'Epoch: {epoch}, Iteration: {i}, Time: {t_comp:.3f}, Data: {t_data:.3f}')
                for loss_name, loss_value in losses.items():
                    print(f'  {loss_name}: {loss_value:.4f}')
                
                # Log losses to TensorBoard
                for loss_name, loss_value in losses.items():
                    writer.add_scalar(f'Loss/{loss_name}', loss_value, total_iterations)
            
            # Display and save generated images
            if total_iterations % opt.display_freq == 0:
                # Get the current visuals
                visuals = {
                    'real_A': model.real_A,
                    'fake_B': model.fake_B,
                    'rec_A': model.rec_A,
                    'real_B': model.real_B,
                    'fake_A': model.fake_A,
                    'rec_B': model.rec_B
                }
                
                # Create a grid of generated images
                img_grid = vutils.make_grid(torch.cat([
                    visuals['real_A'], visuals['fake_B'], visuals['rec_A'],
                    visuals['real_B'], visuals['fake_A'], visuals['rec_B']
                ], dim=0), nrow=opt.batch_size)
                
                # Log images to TensorBoard
                writer.add_image('Images', img_grid, total_iterations)
                
                # Save the images as a PNG file
                if hasattr(opt, 'save_images') and opt.save_images:
                    img_dir = os.path.join(opt.checkpoints_dir, opt.name, 'images')
                    os.makedirs(img_dir, exist_ok=True)
                    vutils.save_image(img_grid, os.path.join(img_dir, f'epoch_{epoch}_iter_{i}.png'))
            
            # Update the learning rate
            if (i + 1) % (len(dataloader) // 10) == 0:
                writer.add_scalar('LR/generator', optimizer_G.param_groups[0]['lr'], total_iterations)
                writer.add_scalar('LR/discriminator', optimizer_D.param_groups[0]['lr'], total_iterations)
            
            # Update timing information
            iter_data_time = time.time()
        
        # Save the model checkpoint
        if epoch % opt.save_epoch_freq == 0:
            print(f'Saving the model at the end of epoch {epoch}')
            model_save_path = os.path.join(opt.checkpoints_dir, opt.name, f'net_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'netG_A': model.netG_A.state_dict(),
                'netG_B': model.netG_B.state_dict(),
                'netD_A': model.netD_A.state_dict(),
                'netD_B': model.netD_B.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict()
            }, model_save_path)
            
            # Save latest model for easy resuming
            latest_save_path = os.path.join(opt.checkpoints_dir, opt.name, 'latest_net.pth')
            torch.save({
                'epoch': epoch,
                'netG_A': model.netG_A.state_dict(),
                'netG_B': model.netG_B.state_dict(),
                'netD_A': model.netD_A.state_dict(),
                'netD_B': model.netD_B.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict()
            }, latest_save_path)
        
        # Update the learning rate at the end of each epoch
        print(f'End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t Time Taken: {time.time() - epoch_start_time:.3f} sec')
        scheduler_G.step()
        scheduler_D.step()
    
    # Close the TensorBoard writer at the end of training
    writer.close()


if __name__ == '__main__':
    main() 