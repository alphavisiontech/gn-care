import torch
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging
from tqdm import tqdm
import time
import wandb
from hw_monitor import ResourceMonitor

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

class CTRGCNTrainer:
    """Trainer class for CTRGCN model with resource monitoring"""
    
    def __init__(self, model, train_loader, val_loader=None, num_classes=2,
                 lr=0.001, weight_decay=1e-4, step_size=50, gamma=0.1,
                 device=None, log_dir='logs', use_wandb=True, 
                 monitor_resources=True, monitor_interval=5.0):
        """
        Args:
            model: CTRGCN model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_classes: Number of classes
            lr: Learning rate
            weight_decay: Weight decay
            step_size: Step size for learning rate scheduler
            gamma: Gamma for learning rate scheduler
            device: Device to use for training
            log_dir: Directory for saving logs
            use_wandb: Whether to use Weights & Biases logging
            monitor_resources: Whether to monitor hardware resources
            monitor_interval: Interval for resource monitoring (seconds)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_wandb = use_wandb
        self.monitor_resources = monitor_resources
        
        # Move model to device
        self.model.to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        
        # Setup logging
        os.makedirs(log_dir, exist_ok=True)
        self.setup_logging(log_dir)
        
        # Setup resource monitoring
        if self.monitor_resources:
            self.resource_monitor = ResourceMonitor(
                monitor_interval=monitor_interval,
                log_to_wandb=self.use_wandb,
                monitor_process_only=True
            )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        # Print initial GPU memory usage
        if torch.cuda.is_available():
            print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
            print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f}GB")
        
    def setup_logging(self, log_dir):
        """Setup logging configuration"""
        log_file = os.path.join(log_dir, f'training_{time.strftime("%Y%m%d_%H%M%S")}.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def get_gpu_memory_info(self):
        """Get detailed GPU memory information"""
        if not torch.cuda.is_available():
            return "N/A"
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        
        return f"{allocated:.1f}GB ({reserved:.1f}GB reserved, {max_allocated:.1f}GB max)"
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Clear GPU cache before starting epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            pred = output.argmax(dim=1)
            total += target.size(0)
            correct += pred.eq(target).sum().item()

            gpu_mem_info = self.get_gpu_memory_info()
            
            # Log batch metrics to wandb
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'batch_loss': loss.item(),
                    'batch_acc': pred.eq(target).float().mean().item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'batch_step': batch_idx,
                    'gpu_memory_allocated_gb': torch.cuda.memory_allocated()/1024**3 if torch.cuda.is_available() else 0,
                    'gpu_memory_reserved_gb': torch.cuda.memory_reserved()/1024**3 if torch.cuda.is_available() else 0,
                })
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'GPU': gpu_mem_info
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total

        # Print CUDA utilization per epoch 
        if torch.cuda.is_available():
            final_allocated = torch.cuda.memory_allocated() / 1024**3
            peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Epoch end - GPU memory: {final_allocated:.2f}GB (peak: {peak_allocated:.2f}GB)")
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """Validate for one epoch"""
        if self.val_loader is None:
            return None, None
            
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                pred = output.argmax(dim=1)
                total += target.size(0)
                correct += pred.eq(target).sum().item()
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%',
                    'GPU': self.get_gpu_memory_info()
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, epochs=100, early_stopping=None, save_dir='checkpoints', start_epoch=0):
        """
        Train the model with resource monitoring
        
        Args:
            epochs: Number of epochs
            early_stopping: EarlyStopping instance
            save_dir: Directory to save model checkpoints
            start_epoch: Starting epoch number (for resume)
        """
        os.makedirs(save_dir, exist_ok=True)
        best_val_acc = 0.0

        # If resuming, find the best validation accuracy from history
        if start_epoch > 0 and self.history.get('val_acc'):
            best_val_acc = max(self.history['val_acc'])
            self.logger.info(f"Resuming training from epoch {start_epoch}")
            self.logger.info(f"Previous best validation accuracy: {best_val_acc:.4f}")

        total_epochs = start_epoch + epochs
        self.logger.info(f"Training from epoch {start_epoch} to {total_epochs}")
        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        self.logger.info(f"Number of classes: {self.num_classes}")
        
        # Start resource monitoring
        if self.monitor_resources:
            self.resource_monitor.start()
        
        try:
            for epoch in range(epochs):
                current_epoch = start_epoch + epoch
                start_time = time.time()
                
                # Clear GPU cache before epoch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Training
                train_loss, train_acc = self.train_epoch()
                
                # Validation
                val_loss, val_acc = self.validate_epoch()
                
                # Learning rate step
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Update history
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['lr'].append(current_lr)
                
                if val_loss is not None:
                    self.history['val_loss'].append(val_loss)
                    self.history['val_acc'].append(val_acc)
                
                # Log to wandb
                if self.use_wandb:
                    wandb_log = {
                        'epoch': current_epoch,
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'learning_rate': current_lr
                    }
                    if val_loss is not None:
                        wandb_log.update({
                            'val_loss': val_loss,
                            'val_acc': val_acc
                        })

                    # Add GPU memory to wandb
                    if torch.cuda.is_available():
                        wandb_log.update({
                            'epoch_gpu_memory_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                            'epoch_gpu_memory_reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                            'epoch_gpu_max_memory_gb': torch.cuda.max_memory_allocated() / 1024**3
                        })
                    
                    wandb.log(wandb_log)
                
                # Logging
                epoch_time = time.time() - start_time
                log_msg = f"Epoch {epoch+1}/{epochs} ({epoch_time:.2f}s) - "
                log_msg += f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                if val_loss is not None:
                    log_msg += f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                log_msg += f", LR: {current_lr:.6f}"
                
                # Add GPU memory info
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.memory_allocated() / 1024**3
                    gpu_max_mem = torch.cuda.max_memory_allocated() / 1024**3
                    log_msg += f", GPU Mem: {gpu_mem:.1f}GB (Max: {gpu_max_mem:.1f}GB)"
                
                self.logger.info(log_msg)
                
                # Save best model
                if val_acc is not None and val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_checkpoint(
                        os.path.join(save_dir, 'best_model.pth'),
                        current_epoch, train_loss, val_loss, val_acc
                    )
                    self.logger.info(f"New best model saved with validation accuracy: {val_acc:.4f}")
                
                # Save latest checkpoint
                if save_dir:
                    self.save_checkpoint(
                        os.path.join(save_dir, f'latest_checkpoint.pth'),
                        current_epoch, train_loss, val_loss, val_acc
                    )
                
                # Early stopping
                if early_stopping is not None:
                    stop_loss = val_loss if val_loss is not None else train_loss
                    if early_stopping(stop_loss, self.model):
                        self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        break
        
        finally:
            # Stop resource monitoring
            if self.monitor_resources:
                self.resource_monitor.stop()
        
        self.logger.info("Training completed!")
        return self.history
    
    def save_checkpoint(self, path, epoch, train_loss, val_loss=None, val_acc=None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'history': self.history,
            'num_classes': self.num_classes
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        return checkpoint
    
    def evaluate(self, test_loader):
        """Evaluate model on test set with resource monitoring"""
        if self.monitor_resources:
            self.resource_monitor.start()
        
        try:
            self.model.eval()
            all_preds = []
            all_targets = []
            test_loss = 0.0
            
            with torch.no_grad():
                pbar = tqdm(test_loader, desc='Testing')
                for data, target in pbar:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    test_loss += loss.item()
                    
                    pred = output.argmax(dim=1)
                    all_preds.extend(pred.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
            
            # Calculate metrics
            test_loss /= len(test_loader)
            test_acc = accuracy_score(all_targets, all_preds)
            
            # Classification report
            if self.num_classes == 2:
                class_names = ['Standing', 'Falling']
            elif self.num_classes == 6:
                class_names = ['Standing', 'Falling', 'Sitting', 'Bending', 'Sleeping', 'Walking']
            else:
                class_names = [f'Class_{i}' for i in range(self.num_classes)]
            
            report = classification_report(all_targets, all_preds, target_names=class_names)
            cm = confusion_matrix(all_targets, all_preds)
            
            # Log test results to wandb
            if self.use_wandb:
                wandb.log({
                    'test_loss': test_loss,
                    'test_acc': test_acc
                })
            
            self.logger.info("Test Results:")
            self.logger.info(f"Test Loss: {test_loss:.4f}")
            self.logger.info(f"Test Accuracy: {test_acc:.4f}")
            self.logger.info(f"Classification Report:\n{report}")
            self.logger.info(f"Confusion Matrix:\n{cm}")
            
            return {
                'test_loss': test_loss,
                'test_acc': test_acc,
                'predictions': all_preds,
                'targets': all_targets,
                'classification_report': report,
                'confusion_matrix': cm
            }
        
        finally:
            if self.monitor_resources:
                self.resource_monitor.stop()