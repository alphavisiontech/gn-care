import argparse
import os
import json
import torch 
from trainer import FallDetectionTrainer
# from kfold_trainer import KFoldFallDetectionTrainer
from tqdm import tqdm
import resource

def parse_args():
    parser = argparse.ArgumentParser(description='K-Fold Cross Validation Training for CTRGCN Fall Detection')
    
    # Data arguments
    parser.add_argument('--dataset_path', nargs='+', 
                        default='/Users/vionna/Desktop/new_fd_train/dataset/v1',
                       help='Path to the COCO dataset directory containing video folders')
    parser.add_argument('--seq_len', type=int, default=30,
                       help='Sequence length for temporal modeling')
    parser.add_argument('--stride', type=int, default=1,
                       help='Stride for sequence sampling')
    
    # Resume training
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Path to checkpoint file to resume training from')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Proportion of data used for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Optimizer arguments
    parser.add_argument('--step_size', type=int, default=10,
                       help='Step size for learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.1,
                       help='Gamma for learning rate scheduler')
    
    # System arguments (optimized for file handle issues)
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, mps)')
    parser.add_argument('--num_workers', type=int, default=2,  # Force multi-threaded
                       help='Number of data loader workers (2 = multi-threaded)')
    
    # Saving arguments
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints and results')
    parser.add_argument('--experiment_name', type=str, default='fall_detection',
                       help='Experiment name for saving')
    
    # Logging arguments
    parser.add_argument('--log_interval', type=int, default=15,
                       help='Log training progress every N batches')
    parser.add_argument('--plot_interval', type=int, default=25,
                       help='Plot confusion matrix every N epochs')
    
    # Wandb
    parser.add_argument('--use_wandb', action='store_true', default=False,
                       help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='fall-detection',
                       help='Weights & Biases project name')
    
    return parser.parse_args()

def setup_device(device_arg):
    """Setup optimal device"""
    if device_arg == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"🖥️  Using device: {device}")
    return device

def optimize_for_stability(config):
    """Optimize configuration for maximum stability"""
    # Reduce batch size if necessary
    if config['batch_size'] > 16:
        print(f"🔧 Reducing batch size from {config['batch_size']} to 16 for stability")
        config['batch_size'] = 16
    
    print("🔧 Applied stability optimizations:")
    print(f"   - num_workers: {config['num_workers']}")
    print(f"   - batch_size: {config['batch_size']}")
    
    return config

def increase_file_limits():
    """Increase file limits"""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"📋 Current file limits - Soft: {soft}, Hard: {hard}")
        
        # Set very conservative limit
        new_soft = min(hard, 256)
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"✅ Updated file limits - Soft: {soft}, Hard: {hard}")
        
    except Exception as e:
        print(f"⚠️  Could not increase file limits: {e}")

def validate_dataset_path(dataset_path):
    """Validate that the dataset path exists and contains video folders"""
    if not os.path.exists(dataset_path):
        raise ValueError(f"❌ Dataset path does not exist: {dataset_path}")
    
    # Check for video folders
    video_folders = [f for f in os.listdir(dataset_path) 
                    if os.path.isdir(os.path.join(dataset_path, f))]
    
    if len(video_folders) == 0:
        raise ValueError(f"❌ No video folders found in dataset path: {dataset_path}")
    
    print(f"✅ Found {len(video_folders)} video folders in dataset")
    
    # Check for annotation files in first few folders
    annotated_folders = 0
    for folder in video_folders[:5]:  # Check first 5 folders
        annotations_path = os.path.join(dataset_path, folder, 'annotations')
        if os.path.exists(annotations_path):
            json_files = [f for f in os.listdir(annotations_path) if f.endswith('.json')]
            if json_files:
                annotated_folders += 1
    
    if annotated_folders == 0:
        raise ValueError(f"❌ No annotation files found in dataset folders")
    
    print(f"✅ Found annotations in {annotated_folders}/5 sampled folders")
    return True

def main():
    args = parse_args()
    
    print(f"\n{'='*80}")
    print(f"🚀 TRAINING CTRGCN")
    print(f"{'='*80}")
    
    # Optimize system
    increase_file_limits()
    
    # Setup device
    device = setup_device(args.device)
    
    # Create configuration
    config = {
        'dataset_path': args.dataset_path,
        'seq_len': args.seq_len,
        'stride': args.stride,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'train_split': args.train_split,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'patience': args.patience,
        'step_size': args.step_size,
        'gamma': args.gamma,
        'device': device,
        'num_workers': args.num_workers, 
        'save_dir': args.save_dir,
        'use_wandb': args.use_wandb,
        'wandb_project': args.wandb_project,
        'experiment_name': args.experiment_name,
        'checkpoint_path': args.checkpoint_path,
        'log_interval': args.log_interval,
        'plot_interval': args.plot_interval,
        'save_interval': args.save_interval,
    }
    
    # Apply stability optimizations
    config = optimize_for_stability(config)
    
    # Create save directory
    print("📁 Setting up directories...")
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['save_dir'], 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(config['save_dir'], 'plots'), exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(config['save_dir'], 'config.json')
    with open(config_path, 'w') as f:
        # Convert device to string for JSON serialization
        config_copy = config.copy()
        config_copy['device'] = str(config['device'])
        json.dump(config_copy, f, indent=2)
    
    print(f"💾 Configuration saved to: {config_path}")
    
    # Check for resume
    if args.checkpoint_path:
        print(f"🔄 Resuming from checkpoint: {args.checkpoint_path}")
    
    # Initialize trainer with checkpoint support
    print("🔧 Initializing Trainer...")
    trainer = FallDetectionTrainer(
        config=config
    )

    try:
        print(f"\n{'='*60}")
        print(f"🎯 STARTING TRAINING")
        print(f"{'='*60}")

        best_accuracy = trainer.train()
        
        print(f"\n🎉 Training completed with best dev accuracy: {best_accuracy:.2f}%")
        print("📈 Final test results are shown above.")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        checkpoint_path = os.path.join(config['save_dir'], 'checkpoints', 'latest_checkpoint.pth')
        if os.path.exists(checkpoint_path):
            print(f"💾 Checkpoint saved. Resume with:")
            print(f"   python train_kfold.py --checkpoint_path {checkpoint_path}")
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        
        # Save emergency checkpoint
        checkpoint_path = os.path.join(config['save_dir'], 'checkpoints', 'latest_checkpoint.pth')
        if os.path.exists(checkpoint_path):
            print(f"💾 Emergency checkpoint available. Resume with:")
            print(f"   python train_kfold.py --checkpoint_path {checkpoint_path}")
        
        import traceback
        traceback.print_exc()
        
    finally:
        if config.get('use_wandb', False):
            try:
                import wandb
                wandb.finish()
                print("📊 Weights & Biases session closed")
            except:
                pass
        # Clean up memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("🧹 Memory cleanup completed")

if __name__ == "__main__":
    main()
