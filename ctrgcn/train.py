import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import json
import numpy as np
import wandb
from ctrgcn import CTRGCN, Graph
from trainer import CTRGCNTrainer, EarlyStopping
from dataset import COCOSkeletonDataset
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train CTRGCN for Fall Detection')
    
    # Data arguments
    parser.add_argument('--train_data_dir', type=str, 
                       default='../dataset/v1',
                       help='Path to training data directory')
    parser.add_argument('--test_data_dir', type=str,
                       default='../dataset/test', 
                       help='Path to test data directory')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes (2 for binary, 8 for full)')
    parser.add_argument('--sequence_length', type=int, default=30,
                       help='Length of input sequences')
    
    # Model arguments
    parser.add_argument('--num_point', type=int, default=17,
                       help='Number of keypoints (COCO: 17)')
    parser.add_argument('--in_channels', type=int, default=3,
                       help='Number of input channels')
    parser.add_argument('--drop_out', type=float, default=0.5,
                       help='Dropout rate')
    parser.add_argument('--adaptive', action='store_true', default=True,
                       help='Use adaptive graph convolution')
    parser.add_argument('--attention', action='store_true', default=True,
                       help='Use attention mechanism')
    
    # Training arguments
    parser.add_argument('--device', type=str, default='auto',
                       help="Device to use ('cuda', 'cpu', or 'auto') (default: auto)")
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--step_size', type=int, default=10,
                       help='Step size for learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.1,
                       help='Gamma for learning rate scheduler')
    
    # Early stopping arguments
    parser.add_argument('--early_stopping', action='store_true', default=True,
                       help='Use early stopping')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=1e-3,
                       help='Minimum delta for early stopping')
    
    # Data loading arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    parser.add_argument('--pin_memory', action='store_true', default=True,
                       help='Pin memory for data loading')
    
    # Validation split
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory to save logs')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory to save results')
    
    # Resume training from checkpoint
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Path to checkpoint model to resume training')
    
    # Test only mode
    parser.add_argument('--test_only', action='store_true', default=False,
                       help='Only run testing with pretrained model')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to pretrained model for testing')
    
    # Wandb and monitoring arguments
    parser.add_argument('--use_wandb', action='store_true', default=True,
                       help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='ctrgcn-fall-detection',
                       help='Wandb project name')
    parser.add_argument('--monitor_resources', action='store_true', default=True,
                       help='Monitor hardware resources during training')
    parser.add_argument('--monitor_interval', type=float, default=5.0,
                       help='Resource monitoring interval in seconds')
    
    return parser.parse_args()

def get_device(device_arg):
    """Get the appropriate device"""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"CUDA is available. Using device: {device}")
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = torch.device('cpu')
            print("CUDA is not available. Using CPU.")
    else:
        device = torch.device(device_arg)
        print(f"Using specified device: {device}")

def setup_wandb(args):
    """Initialize Weights & Biases"""
    if not args.use_wandb:
        return
    
    # Create wandb config
    config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'sequence_length': args.sequence_length,
        'num_classes': args.num_classes,
        'drop_out': args.drop_out,
        'adaptive': args.adaptive,
        'attention': args.attention,
        'step_size': args.step_size,
        'gamma': args.gamma,
        'num_point': args.num_point,
        'in_channels': args.in_channels,
        'architecture': 'CTRGCN'
    }
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        config=config,
        save_code=True
    )
    
    print(f"Wandb initialized. Project: {args.wandb_project}")

def create_data_loaders(args):
    """Create data loaders for training and testing"""
    
    # Training dataset
    print(f"Loading training data from: {args.train_data_dir}")
    train_dataset = COCOSkeletonDataset(
        data_dir=args.train_data_dir,
        sequence_length=args.sequence_length,
        num_classes=args.num_classes,
        class_mapping=None  # Let dataset discover classes
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    train_dataset.print_class_info()
    
    # Get the class mapping from training dataset
    class_mapping = train_dataset.get_class_mapping()
    
    # Split training data for validation if needed
    if args.val_split > 0 and not args.test_only:
        val_size = int(len(train_dataset) * args.val_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        print(f"Training split size: {len(train_dataset)}")
        print(f"Validation split size: {len(val_dataset)}")
    else:
        val_dataset = None
    
    # Test dataset (use same class mapping as training)
    print(f"Loading test data from: {args.test_data_dir}")
    test_dataset = COCOSkeletonDataset(
        data_dir=args.test_data_dir,
        sequence_length=args.sequence_length,
        num_classes=args.num_classes,
        class_mapping=class_mapping  # Use training class mapping
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    ) if not args.test_only else None
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    ) if val_dataset is not None else None
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    
    return train_loader, val_loader, test_loader, class_mapping

def create_model(args):
    """Create CTRGCN model"""
    # Create graph
    graph = Graph(layout='coco', strategy='spatial', max_hop=1)
    
    # Create model
    model = CTRGCN(
        num_class=args.num_classes,
        num_point=args.num_point,
        num_person=1,
        graph=graph,
        in_channels=args.in_channels,
        drop_out=args.drop_out,
        adaptive=args.adaptive,
        attention=args.attention
    )
    
    return model

def plot_training_history(history, save_path):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history and history['val_loss']:
        axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy plot
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy')
    if 'val_acc' in history and history['val_acc']:
        axes[0, 1].plot(history['val_acc'], label='Validation Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning rate plot
    axes[1, 0].plot(history['lr'], label='Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Remove empty subplot
    fig.delaxes(axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm, class_names, save_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    args = parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Set device
    device = get_device(args.device)
    
    # Setup wandb
    if args.use_wandb and not args.test_only:
        setup_wandb(args)
    
    # Create data loaders
    train_loader, val_loader, test_loader, class_mapping = create_data_loaders(args)
    
    # Create model
    model = create_model(args)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Log model to wandb
    if args.use_wandb:
        wandb.watch(model, log_freq=100)
    
    if args.test_only:
        # Test only mode
        if args.model_path is None:
            raise ValueError("model_path must be specified for test-only mode")
        
        print(f"Loading model from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create trainer for evaluation
        trainer = CTRGCNTrainer(
            model=model,
            train_loader=None,
            val_loader=None,
            num_classes=args.num_classes,
            device=device,
            log_dir=args.log_dir,
            use_wandb=args.use_wandb,
            monitor_resources=args.monitor_resources,
            monitor_interval=args.monitor_interval
        )
        
        # Evaluate on test set
        print("Evaluating on test set...")
        results = trainer.evaluate(test_loader)
        
        # Save results
        results_file = os.path.join(args.results_dir, 'test_results.json')
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {
                'test_loss': results['test_loss'],
                'test_acc': results['test_acc'],
                'classification_report': results['classification_report'],
                'confusion_matrix': results['confusion_matrix'].tolist(),
                'class_mapping': class_mapping
            }
            json.dump(json_results, f, indent=2)
        
        # Plot confusion matrix
        if args.num_classes == 2:
            class_names = ['Standing', 'Falling']
        elif args.num_classes == 6:
            class_names = ['Standing', 'Falling', 'Sitting', 'Bending', 'Sleeping', 'Walking']
        else:
            class_names = [name for name, _ in sorted(class_mapping.items(), key=lambda x: x[1])]
        
        cm_plot_path = os.path.join(args.results_dir, 'confusion_matrix.png')
        plot_confusion_matrix(results['confusion_matrix'], class_names, cm_plot_path)
        
        print(f"Test results saved to {results_file}")
        print(f"Confusion matrix plot saved to {cm_plot_path}")
        
        if args.use_wandb:
            wandb.finish()
        
        return
    
    # Create trainer
    trainer = CTRGCNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=args.num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        step_size=args.step_size,
        gamma=args.gamma,
        device=device,
        log_dir=args.log_dir,
        use_wandb=args.use_wandb,
        monitor_resources=args.monitor_resources,
        monitor_interval=args.monitor_interval
    )
    
    # Resume training from checkppoint if specified
    start_epoch = 0
    if args.checkpoint_path is not None:
        print(f"Resuming training from {args.checkpoint_path}")
        checkpoint = trainer.load_checkpoint(args.checkpoint_path)
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    # Calculate remaining epochs
    remaining_epochs = max(0, args.epochs - start_epoch)
    if remaining_epochs == 0:
        print(f"Warning: No remaining epochs to train (start_epoch={start_epoch}, total_epochs={args.epochs})")
    
    # Early stopping
    early_stopping = None
    if args.early_stopping:
        early_stopping = EarlyStopping(
            patience=args.patience,
            min_delta=args.min_delta,
            restore_best_weights=True
        )
        print(f"Early stopping enabled with patience={args.patience}")
    
    # Train model
    print("Starting training...")
    history = trainer.train(
        epochs=remaining_epochs,
        early_stopping=early_stopping,
        save_dir=args.save_dir,
        start_epoch=start_epoch
    )
    
    # Save final model with class mapping
    final_model_path = os.path.join(args.save_dir, 'final_model.pth')
    trainer.save_checkpoint(final_model_path, args.epochs-1, 
                           history['train_loss'][-1], 
                           history['val_loss'][-1] if history['val_loss'] else None,
                           history['val_acc'][-1] if history['val_acc'] else None)
    
    # Save class mapping
    class_mapping_file = os.path.join(args.save_dir, 'class_mapping.json')
    with open(class_mapping_file, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    # Plot training history
    history_plot_path = os.path.join(args.results_dir, 'training_history.png')
    plot_training_history(history, history_plot_path)
    print(f"Training history plot saved to {history_plot_path}")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    results = trainer.evaluate(test_loader)
    
    # Save results
    results_file = os.path.join(args.results_dir, 'test_results.json')
    with open(results_file, 'w') as f:
        json_results = {
            'test_loss': results['test_loss'],
            'test_acc': results['test_acc'],
            'classification_report': results['classification_report'],
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'training_history': history,
            'class_mapping': class_mapping
        }
        json.dump(json_results, f, indent=2)
    
    # Plot confusion matrix
    if args.num_classes == 2:
        class_names = ['Standing', 'Falling']
    else:
        class_names = [name for name, _ in sorted(class_mapping.items(), key=lambda x: x[1])]
    
    cm_plot_path = os.path.join(args.results_dir, 'confusion_matrix.png')
    plot_confusion_matrix(results['confusion_matrix'], class_names, cm_plot_path)
    
    print(f"Training completed!")
    print(f"Best model saved to {os.path.join(args.save_dir, 'best_model.pth')}")
    print(f"Final model saved to {final_model_path}")
    print(f"Class mapping saved to {class_mapping_file}")
    print(f"Results saved to {results_file}")
    print(f"Plots saved to {args.results_dir}")
    
    print(f"\nFinal Test Results:")
    print(f"Test Accuracy: {results['test_acc']:.4f}")
    print(f"Test Loss: {results['test_loss']:.4f}")
    
    # Finish wandb run
    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()