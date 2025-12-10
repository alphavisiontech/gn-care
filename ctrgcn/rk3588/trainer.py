import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange
from dataset import create_data_loaders
from ctrgcn_rk3588 import CTRGCN_RK3588

class EarlyStopping:
    """Early stopping utility to stop training when validation loss stops improving"""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True, verbose=True):
        """
        Args:
            patience (int): How many epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as an improvement
            restore_best_weights (bool): Whether to restore best weights when stopping
            verbose (bool): Whether to print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, val_loss, val_accuracy, model):
        """
        Call early stopping check
        
        Args:
            val_loss: Current validation loss
            val_accuracy: Current validation accuracy  
            model: Model to save weights from
        """
        # Check if this is the best model so far (using validation loss as primary metric)
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_accuracy = val_accuracy
            self.counter = 0
            
            # Save best weights
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            
            if self.verbose:
                print(f"✅ Early stopping: New best validation loss: {val_loss:.4f}")
                
        else:
            self.counter += 1
            if self.verbose:
                print(f"⏳ Early stopping: {self.counter}/{self.patience} - No improvement in validation loss")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"🛑 Early stopping triggered! No improvement for {self.patience} epochs")
                
                # Restore best weights if requested
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    if self.verbose:
                        print(f"🔄 Restored best weights (val_loss: {self.best_loss:.4f}, val_acc: {self.best_accuracy:.2f}%)")

class FallDetectionTrainer:
    def __init__(self, config):
        self.config = config
        # M4 device optimization
        if config.get('device') == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(config['device'])
        
        self.best_accuracy = 0.0
        self.start_epoch = 0
        
        # Early stopping setup
        self.early_stopping = EarlyStopping(
            patience=config.get('patience', 10),
            min_delta=config.get('min_delta', 0.001),
            restore_best_weights=config.get('restore_weights', True),
            verbose=True
        )
        
        # Create save directory
        os.makedirs(config['save_dir'], exist_ok=True)
        os.makedirs(os.path.join(config['save_dir'], 'plots'), exist_ok=True)
        
        # Initialize wandb if enabled
        if config['use_wandb']:
            wandb.init(
                project=config['wandb_project'],
                config=config,
                name=config['experiment_name']
            )
        
        print(f"🚀 Using device: {self.device}")
        print(f"⏱️  Early stopping enabled with patience: {self.early_stopping.patience}")
        
        # Initialize data loaders
        self._setup_data_loaders()
        
        # Initialize model
        self._setup_model()
        
        # Initialize optimizer and scheduler
        self._setup_optimizer()
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Resume from checkpoint if specified
        if config.get('checkpoint_path'):
            self._load_checkpoint(config['checkpoint_path'])
    
    def _setup_data_loaders(self):
        """Setup train and validation data loaders with RK3588 optimizations"""
        print("📂 Loading data...")
        
        # Add progress bar for data loading
        with tqdm(total=100, desc="Loading dataset", unit="%") as pbar:
            pbar.update(20)
            
            # Note: create_data_loaders returns (train_loader, val_loader, num_classes)
            self.train_loader, self.dev_loader, self.num_classes = create_data_loaders(
                self.config['dataset_path'], 
                batch_size=self.config['batch_size'],
                seq_len=self.config['seq_len'],
                stride=self.config.get('stride', 1),
                train_split=self.config.get('train_split', 0.8),
                num_workers=self.config.get('num_workers', 1),
                label_strategy=self.config.get('label_strategy', 'transition')
            )
            pbar.update(60)
            
            # For this implementation, we'll use dev_loader as both dev and test
            # In a real scenario, you'd want separate test data
            self.test_loader = self.dev_loader
            pbar.update(20)
        
        print(f"✅ Data loaded - Train: {len(self.train_loader.dataset)}, "
              f"Dev: {len(self.dev_loader.dataset)}")
        print(f"📊 Number of classes: {self.num_classes}")
        
        # Update config with actual num_classes from dataset
        self.config['num_classes'] = self.num_classes

    def _setup_model(self):
        """Initialize the CTRGCN model with RK3588 optimizations"""
        print("🏗️  Setting up model...")
        with tqdm(total=100, desc="Initializing model", unit="%") as pbar:
            pbar.update(30)
            
            self.model = CTRGCN_RK3588(
                num_classes=self.num_classes,  # Use actual number of classes from dataset
                num_joints=17, 
                in_channels=3,
                base_channel=self.config.get('base_channel', 16),
                drop_out=self.config.get('drop_out', 0.3)
            )
            pbar.update(40)
            
            self.model = self.model.to(self.device)
            pbar.update(30)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"📊 Model: {total_params:,} total params, {trainable_params:,} trainable")
        print(f"🎯 Model configured for {self.num_classes} classes")
        
        if self.config['use_wandb']:
            wandb.watch(self.model, log="all")
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler"""
        print("⚙️  Setting up optimizer...")
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.config['step_size'], 
            gamma=self.config['gamma']
        )
    
    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint for resuming training"""
        if os.path.exists(checkpoint_path):
            print(f"📁 Loading checkpoint from {checkpoint_path}")
            with tqdm(total=100, desc="Loading checkpoint", unit="%") as pbar:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                pbar.update(25)
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                pbar.update(25)
                
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                pbar.update(25)
                
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_accuracy = checkpoint['best_accuracy']
                
                # Restore early stopping state if available
                if 'early_stopping_state' in checkpoint:
                    early_state = checkpoint['early_stopping_state']
                    self.early_stopping.best_loss = early_state.get('best_loss', float('inf'))
                    self.early_stopping.best_accuracy = early_state.get('best_accuracy', 0.0)
                    self.early_stopping.counter = early_state.get('counter', 0)
                pbar.update(25)
            
            print(f"✅ Resumed from epoch {self.start_epoch}, best accuracy: {self.best_accuracy:.2f}%")
        else:
            print(f"❌ Checkpoint {checkpoint_path} not found. Starting training from scratch.")
    
    def _save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint with early stopping state"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_accuracy': self.best_accuracy,
            'config': self.config,
            'early_stopping_state': {
                'best_loss': self.early_stopping.best_loss,
                'best_accuracy': self.early_stopping.best_accuracy,
                'counter': self.early_stopping.counter
            }
        }
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config['save_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f'💾 New best model saved with accuracy: {self.best_accuracy:.2f}%')
        
        # Save latest checkpoint
        latest_path = os.path.join(self.config['save_dir'], 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
    
    def _train_epoch(self, epoch):
        """Train for one epoch with progress bar"""
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Create progress bar for training batches
        train_pbar = tqdm(
            self.train_loader, 
            desc=f"🏋️  Training Epoch {epoch+1}/{self.config['num_epochs']}", 
            leave=False,
            unit="batch"
        )
        
        for batch_idx, (data, labels) in enumerate(train_pbar):
            data = data.to(self.device, non_blocking=True)
            
            # Convert labels to LongTensor and ensure 1D
            if isinstance(labels, (list, tuple)):
                labels = torch.LongTensor(labels)
            elif not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long)
            else:
                labels = labels.long()
            
            # Ensure labels are 1D
            if labels.dim() > 1:
                labels = labels.squeeze()
            
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar with current metrics
            current_acc = 100 * train_correct / train_total if train_total > 0 else 0
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        train_accuracy = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(self.train_loader)
        
        # Clean up MPS cache if using Apple Silicon
        if self.device.type == 'mps':
            torch.mps.empty_cache()
        
        return avg_train_loss, train_accuracy
    
    def _validate_epoch(self, data_loader, split_name="dev", epoch=None):
        """Validate the model on given data loader with progress bar"""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_labels = []
        
        # Create progress bar for validation
        desc = f"🧪 Validating ({split_name})"
        if epoch is not None:
            desc += f" Epoch {epoch+1}"
        
        val_pbar = tqdm(
            data_loader, 
            desc=desc,
            leave=False,
            unit="batch"
        )
        
        with torch.no_grad():
            for data, labels in val_pbar:
                data = data.to(self.device, non_blocking=True)
                
                # Convert labels to LongTensor and ensure 1D
                if isinstance(labels, (list, tuple)):
                    labels = torch.LongTensor(labels)
                elif not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels, dtype=torch.long)
                else:
                    labels = labels.long()
                
                # Ensure labels are 1D
                if labels.dim() > 1:
                    labels = labels.squeeze()
                
                labels = labels.to(self.device, non_blocking=True)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                current_acc = 100 * val_correct / val_total if val_total > 0 else 0
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
        
        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(data_loader)
        
        # Clean up MPS cache if using Apple Silicon
        if self.device.type == 'mps':
            torch.mps.empty_cache()
        
        return avg_val_loss, val_accuracy, all_predictions, all_labels
    
    def _get_target_names_and_labels(self, y_true, y_pred):
        """Get target names and labels based on actual classes present in data"""
        # Get unique classes present in the data
        unique_classes = sorted(set(y_true) | set(y_pred))
        
        # Define all possible class names
        all_class_names = {
            0: 'walking',
            1: 'sitting', 
            2: 'standing',
            3: 'falling',
            4: 'bending_down',
            5: 'crouching',
            6: 'waving_hands',
            7: 'sleeping'
        }
        
        # Create target names only for classes present in data
        target_names = [all_class_names.get(i, f'class_{i}') for i in unique_classes]
        
        return target_names, unique_classes
    
    def _plot_confusion_matrix(self, y_true, y_pred, epoch, split_name="dev"):
        """Plot and save confusion matrix"""
        # Get target names for actual classes in data
        target_names, unique_labels = self._get_target_names_and_labels(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        
        # Create confusion matrix with proper labels
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
        plt.figure(figsize=(12, 10))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names,
                   yticklabels=target_names)
        plt.title(f'Confusion Matrix ({split_name}) - Epoch {epoch+1}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plot_path = os.path.join(self.config['save_dir'], 'plots', f'confusion_matrix_{split_name}_epoch_{epoch+1}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.config['use_wandb']:
            wandb.log({f"confusion_matrix_{split_name}_epoch_{epoch+1}": wandb.Image(plot_path)})
    
    def _plot_training_curves(self, train_losses, train_accuracies, dev_losses, dev_accuracies):
        """Plot and save training curves"""
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, dev_losses, 'r-', label='Dev Loss')
        ax1.set_title('Training and Dev Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(epochs, train_accuracies, 'b-', label='Train Accuracy')
        ax2.plot(epochs, dev_accuracies, 'r-', label='Dev Accuracy')
        ax2.set_title('Training and Dev Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plot_path = os.path.join(self.config['save_dir'], 'plots', 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.config['use_wandb']:
            wandb.log({"training_curves": wandb.Image(plot_path)})
    
    def train(self):
        """Main training loop with early stopping and progress bars"""
        print("🏃‍♂️ Starting training...")
        
        train_losses = []
        train_accuracies = []
        dev_losses = []
        dev_accuracies = []
        
        # Main epoch progress bar
        epoch_pbar = trange(
            self.start_epoch, 
            self.config['num_epochs'], 
            desc="🎯 Training Progress",
            unit="epoch"
        )
        
        for epoch in epoch_pbar:
            # Training phase
            train_loss, train_accuracy = self._train_epoch(epoch)
            
            # Development validation phase
            dev_loss, dev_accuracy, dev_predictions, dev_labels = self._validate_epoch(
                self.dev_loader, "dev", epoch
            )
            
            # Update learning rate
            self.scheduler.step()
            
            # Store metrics
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            dev_losses.append(dev_loss)
            dev_accuracies.append(dev_accuracy)
            
            # Log metrics
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'dev_loss': dev_loss,
                'dev_accuracy': dev_accuracy,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'early_stopping_counter': self.early_stopping.counter
            }
            
            if self.config['use_wandb']:
                wandb.log(metrics)
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'Train Acc': f'{train_accuracy:.2f}%',
                'Dev Acc': f'{dev_accuracy:.2f}%',
                'Best': f'{self.best_accuracy:.2f}%',
                'ES': f'{self.early_stopping.counter}/{self.early_stopping.patience}'
            })
            
            # Check for best model (using dev accuracy for model selection)
            is_best = dev_accuracy > self.best_accuracy
            if is_best:
                self.best_accuracy = dev_accuracy
            
            # Early stopping check (using dev loss)
            self.early_stopping(dev_loss, dev_accuracy, self.model)
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_interval', 10) == 0 or is_best:
                self._save_checkpoint(epoch, is_best)
            
            # Plot confusion matrix periodically
            if (epoch + 1) % self.config.get('plot_interval', 25) == 0:
                print(f"\n📊 Generating confusion matrix for epoch {epoch+1}...")
                self._plot_confusion_matrix(dev_labels, dev_predictions, epoch, "dev")
            
            # Check if early stopping was triggered
            if self.early_stopping.early_stop:
                epoch_pbar.set_description("🛑 Early Stopping")
                epoch_pbar.close()
                print(f"\n🛑 Training stopped early at epoch {epoch + 1}")
                print(f"💡 Best validation loss: {self.early_stopping.best_loss:.4f}")
                print(f"💡 Best validation accuracy: {self.early_stopping.best_accuracy:.2f}%")
                break
        
        # Final evaluation on test set
        final_epoch = epoch if self.early_stopping.early_stop else self.config['num_epochs'] - 1
        print(f'\n🎉 Training completed!')
        print(f'📊 Total epochs: {final_epoch + 1}')
        print(f'🏆 Best dev accuracy: {self.best_accuracy:.2f}%')
        
        # Test the best model
        print("\n🧪 Evaluating best model on test set...")
        best_model_path = os.path.join(self.config['save_dir'], 'best_model.pth')
        if os.path.exists(best_model_path):
            with tqdm(total=100, desc="Loading best model", unit="%") as pbar:
                checkpoint = torch.load(best_model_path, map_location=self.device)
                pbar.update(50)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                pbar.update(50)
            
            test_loss, test_accuracy, test_predictions, test_labels = self._validate_epoch(
                self.test_loader, "test"
            )
            print(f'🧪 Final test accuracy: {test_accuracy:.2f}%')
            
            # Generate final plots and reports
            print("\n📊 Generating final plots and reports...")
            with tqdm(total=100, desc="Generating reports", unit="%") as pbar:
                # Final confusion matrices
                self._plot_confusion_matrix(dev_labels, dev_predictions, final_epoch, "dev")
                pbar.update(25)
                
                self._plot_confusion_matrix(test_labels, test_predictions, final_epoch, "test")
                pbar.update(25)
                
                # Training curves
                self._plot_training_curves(train_losses, train_accuracies, dev_losses, dev_accuracies)
                pbar.update(25)
                
                # Get target names for actual classes present in data
                dev_target_names, dev_unique_labels = self._get_target_names_and_labels(dev_labels, dev_predictions)
                test_target_names, test_unique_labels = self._get_target_names_and_labels(test_labels, test_predictions)
                
                # Generate classification reports with proper labels
                dev_report = classification_report(
                    dev_labels, dev_predictions, 
                    labels=dev_unique_labels,
                    target_names=dev_target_names, 
                    output_dict=True,
                    zero_division=0
                )
                test_report = classification_report(
                    test_labels, test_predictions, 
                    labels=test_unique_labels,
                    target_names=test_target_names, 
                    output_dict=True,
                    zero_division=0
                )
                pbar.update(25)
            
            # Print final reports
            print('\n📈 Dev Set Classification Report:')
            print(classification_report(dev_labels, dev_predictions, 
                                      labels=dev_unique_labels,
                                      target_names=dev_target_names,
                                      zero_division=0))
            
            print('\n📈 Test Set Classification Report:')
            print(classification_report(test_labels, test_predictions, 
                                      labels=test_unique_labels,
                                      target_names=test_target_names,
                                      zero_division=0))
            
            if self.config['use_wandb']:
                # Log final metrics
                wandb_metrics = {
                    'final_dev_accuracy': self.best_accuracy,
                    'final_test_accuracy': test_accuracy,
                    'total_epochs': final_epoch + 1,
                    'early_stopped': self.early_stopping.early_stop,
                    'num_classes_dev': len(dev_unique_labels),
                    'num_classes_test': len(test_unique_labels),
                    'classes_present_dev': dev_unique_labels,
                    'classes_present_test': test_unique_labels
                }
                
                # Add falling-specific metrics if falling class is present
                if 'falling' in dev_target_names:
                    falling_idx_dev = dev_target_names.index('falling')
                    actual_falling_label = dev_unique_labels[falling_idx_dev]
                    if str(actual_falling_label) in dev_report:
                        wandb_metrics.update({
                            'final_dev_precision_falling': dev_report[str(actual_falling_label)]['precision'],
                            'final_dev_recall_falling': dev_report[str(actual_falling_label)]['recall'],
                            'final_dev_f1_falling': dev_report[str(actual_falling_label)]['f1-score'],
                        })
                
                if 'falling' in test_target_names:
                    falling_idx_test = test_target_names.index('falling')
                    actual_falling_label = test_unique_labels[falling_idx_test]
                    if str(actual_falling_label) in test_report:
                        wandb_metrics.update({
                            'final_test_precision_falling': test_report[str(actual_falling_label)]['precision'],
                            'final_test_recall_falling': test_report[str(actual_falling_label)]['recall'],
                            'final_test_f1_falling': test_report[str(actual_falling_label)]['f1-score']
                        })
                
                wandb.log(wandb_metrics)
                wandb.finish()
        
        return self.best_accuracy

def test_model(checkpoint_path, dataset_path, config):
    """Test a trained model on a complete different dataset with progress bars"""
    device = torch.device(config.get('device', 'cpu'))
    if device.type == 'mps' and not torch.backends.mps.is_available():
        device = torch.device('cpu')
    
    print("🧪 Loading model for testing...")
    
    # Load checkpoint with progress
    with tqdm(total=100, desc="Loading checkpoint", unit="%") as pbar:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        saved_config = checkpoint['config']
        pbar.update(50)
        
        # Create test loader
        _, _, test_loader = create_data_loaders(
            dataset_path, 
            batch_size=config.get('batch_size', saved_config['batch_size']),
            seq_len=saved_config['seq_len'],
            stride=saved_config.get('stride', 1),
            num_workers=config.get('num_workers', 4),
            label_strategy=saved_config.get('label_strategy', 'transition')
        )
        pbar.update(30)
        
        # Initialize and load model
        model = CTRGCN_RK3588(
            num_classes=saved_config['num_classes'], 
            num_joints=saved_config['num_joints'], 
            in_channels=saved_config['in_channels']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        pbar.update(20)
    
    # Test with progress bar
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    test_pbar = tqdm(test_loader, desc="🧪 Testing model", unit="batch")
    
    with torch.no_grad():
        for data, labels in test_pbar:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            current_acc = 100 * correct / total if total > 0 else 0
            test_pbar.set_postfix({'Accuracy': f'{current_acc:.2f}%'})
    
    accuracy = 100 * correct / total
    print(f'🧪 Test Accuracy: {accuracy:.2f}%')

    # Get proper target names and labels
    unique_classes = sorted(set(all_labels) | set(all_predictions))
    all_class_names = {
        0: 'walking', 1: 'sitting', 2: 'standing', 3: 'falling',
        4: 'bending_down', 5: 'crouching', 6: 'waving_hands', 7: 'sleeping'
    }
    target_names = [all_class_names.get(i, f'class_{i}') for i in unique_classes]

    print('\n📈 Classification Report:')
    print(classification_report(
        all_labels, all_predictions, 
        labels=unique_classes,
        target_names=target_names,
        zero_division=0
    ))
    
    return accuracy, all_predictions, all_labels