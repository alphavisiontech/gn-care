import psutil
import time
import threading
import torch
import platform
import sys
import os

# Handle wandb import gracefully
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging to wandb disabled.")

# Handle nvidia-ml-py import gracefully
try:
    import pynvml as nvml
    NVML_AVAILABLE = True
    nvml.nvmlInit()
    print("nvidia-ml-py initialized successfully")
except ImportError:
    NVML_AVAILABLE = False
    print("Info: nvidia-ml-py not available. Advanced GPU monitoring disabled.")
except Exception as e:
    NVML_AVAILABLE = False
    print(f"Info: nvidia-ml-py initialization failed: {e}. Advanced GPU monitoring disabled.")

class ResourceMonitor:
    """Hardware resource monitor focused on training process with wandb integration"""
    
    def __init__(self, monitor_interval=1.0, log_to_wandb=True, monitor_process_only=True):
        self.monitor_interval = monitor_interval
        self.log_to_wandb = log_to_wandb and WANDB_AVAILABLE
        self.monitoring = False
        self.thread = None
        self.gpu_available = torch.cuda.is_available()
        self.nvml_available = NVML_AVAILABLE and self.gpu_available
        self.system_platform = platform.system()
        self.monitor_process_only = monitor_process_only
        
        # Get current process for process-specific monitoring
        self.current_process = psutil.Process(os.getpid())
        
        # Baseline measurements (before training starts)
        self.baseline_memory = None
        self.baseline_gpu_memory = None
        self.peak_process_memory = 0
        self.peak_gpu_memory = 0
        
        if self.gpu_available:
            self.gpu_count = torch.cuda.device_count()
            gpu_info = "basic PyTorch" if not self.nvml_available else "advanced NVML"
            print(f"GPU monitoring enabled for {self.gpu_count} GPU(s) with {gpu_info} support")
            
            # Set baseline GPU memory
            self.baseline_gpu_memory = {i: torch.cuda.memory_allocated(i) / (1024**3) 
                                      for i in range(self.gpu_count)}
        else:
            self.gpu_count = 0
            print("GPU monitoring disabled (no CUDA available)")
        
        # Set baseline process memory
        try:
            self.baseline_memory = self.current_process.memory_info().rss / (1024**3)
            print(f"Baseline process memory: {self.baseline_memory:.2f}GB")
        except Exception:
            self.baseline_memory = 0
        
        print(f"System platform: {self.system_platform}")
        print(f"Process-specific monitoring: {self.monitor_process_only}")
        if not WANDB_AVAILABLE and log_to_wandb:
            print("Warning: wandb logging requested but not available")
    
    def get_process_cpu_info(self):
        """Get CPU usage for current process"""
        try:
            # Get process-specific CPU usage
            process_cpu = self.current_process.cpu_percent(interval=None)
            
            # Also get system-wide for context
            system_cpu = psutil.cpu_percent(interval=None)
            
            return {
                'process_cpu_percent': process_cpu,
                'system_cpu_percent': system_cpu,
                'cpu_count': psutil.cpu_count()
            }
        except Exception as e:
            print(f"Error getting process CPU info: {e}")
            return {
                'process_cpu_percent': 0,
                'system_cpu_percent': 0,
                'cpu_count': 0
            }
    
    def get_process_memory_info(self):
        """Get memory usage for current process"""
        try:
            # Process memory info
            memory_info = self.current_process.memory_info()
            process_memory_gb = memory_info.rss / (1024**3)  # Resident Set Size
            process_virtual_gb = memory_info.vms / (1024**3)  # Virtual Memory Size
            
            # Calculate memory used by training (above baseline)
            training_memory_gb = max(0, process_memory_gb - self.baseline_memory)
            
            # Update peak
            self.peak_process_memory = max(self.peak_process_memory, process_memory_gb)
            
            # System memory for context
            system_memory = psutil.virtual_memory()
            system_memory_gb = system_memory.total / (1024**3)
            
            # Calculate process percentage of system memory
            process_memory_percent = (process_memory_gb / system_memory_gb * 100) if system_memory_gb > 0 else 0
            
            return {
                'process_memory_gb': process_memory_gb,
                'process_memory_percent': process_memory_percent,
                'training_memory_gb': training_memory_gb,
                'peak_process_memory_gb': self.peak_process_memory,
                'process_virtual_gb': process_virtual_gb,
                'system_memory_total_gb': system_memory_gb,
                'system_memory_available_gb': system_memory.available / (1024**3)
            }
        except Exception as e:
            print(f"Error getting process memory info: {e}")
            return {
                'process_memory_gb': 0,
                'process_memory_percent': 0,
                'training_memory_gb': 0,
                'peak_process_memory_gb': 0,
                'process_virtual_gb': 0,
                'system_memory_total_gb': 0,
                'system_memory_available_gb': 0
            }
    
    def get_training_gpu_info(self):
        """Get GPU memory usage specific to training"""
        if not self.gpu_available:
            return {}
        
        gpu_info = {}
        try:
            for i in range(self.gpu_count):
                # Current PyTorch memory usage
                current_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                current_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                max_allocated = torch.cuda.max_memory_allocated(i) / (1024**3)
                
                # Calculate training-specific GPU memory (above baseline)
                baseline = self.baseline_gpu_memory.get(i, 0)
                training_gpu_memory = max(0, current_allocated - baseline)
                
                # Update peak
                if i not in self.peak_gpu_memory or isinstance(self.peak_gpu_memory, (int, float)):
                    if isinstance(self.peak_gpu_memory, (int, float)):
                        self.peak_gpu_memory = {}
                    self.peak_gpu_memory[i] = current_allocated
                else:
                    self.peak_gpu_memory[i] = max(self.peak_gpu_memory[i], current_allocated)
                
                # Get total GPU memory
                if self.nvml_available:
                    try:
                        handle = nvml.nvmlDeviceGetHandleByIndex(i)
                        memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                        total_memory = memory_info.total / (1024**3)
                        
                        # Get GPU utilization
                        utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_util = utilization.gpu
                        
                        # Get temperature
                        temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                        
                        gpu_info[f'gpu_{i}'] = {
                            'name': torch.cuda.get_device_name(i),
                            'utilization_percent': gpu_util,
                            'allocated_gb': current_allocated,
                            'reserved_gb': current_reserved,
                            'total_memory_gb': total_memory,
                            'memory_percent': (current_allocated / total_memory * 100) if total_memory > 0 else 0,
                            'training_memory_gb': training_gpu_memory,
                            'peak_allocated_gb': self.peak_gpu_memory[i],
                            'max_allocated_gb': max_allocated,
                            'temperature_c': temp
                        }
                    except Exception as e:
                        # Fallback to basic info
                        props = torch.cuda.get_device_properties(i)
                        total_memory = props.total_memory / (1024**3)
                        
                        gpu_info[f'gpu_{i}'] = {
                            'name': props.name,
                            'allocated_gb': current_allocated,
                            'reserved_gb': current_reserved,
                            'total_memory_gb': total_memory,
                            'memory_percent': (current_allocated / total_memory * 100) if total_memory > 0 else 0,
                            'training_memory_gb': training_gpu_memory,
                            'peak_allocated_gb': self.peak_gpu_memory[i],
                            'max_allocated_gb': max_allocated
                        }
                else:
                    # Basic PyTorch info only
                    props = torch.cuda.get_device_properties(i)
                    total_memory = props.total_memory / (1024**3)
                    
                    gpu_info[f'gpu_{i}'] = {
                        'name': props.name,
                        'allocated_gb': current_allocated,
                        'reserved_gb': current_reserved,
                        'total_memory_gb': total_memory,
                        'memory_percent': (current_allocated / total_memory * 100) if total_memory > 0 else 0,
                        'training_memory_gb': training_gpu_memory,
                        'peak_allocated_gb': self.peak_gpu_memory[i] if isinstance(self.peak_gpu_memory, dict) else current_allocated,
                        'max_allocated_gb': max_allocated
                    }
        except Exception as e:
            print(f"Error getting training GPU info: {e}")
        
        return gpu_info
    
    def get_all_metrics(self):
        """Get all training-specific metrics"""
        metrics = {}
        
        # Get process-specific metrics
        try:
            if self.monitor_process_only:
                metrics.update(self.get_process_cpu_info())
                metrics.update(self.get_process_memory_info())
            else:
                # Fallback to system metrics
                cpu_info = {'cpu_percent': psutil.cpu_percent(), 'cpu_count': psutil.cpu_count()}
                memory = psutil.virtual_memory()
                memory_info = {
                    'memory_percent': memory.percent,
                    'memory_used_gb': memory.used / (1024**3),
                    'memory_total_gb': memory.total / (1024**3),
                    'memory_available_gb': memory.available / (1024**3)
                }
                metrics.update(cpu_info)
                metrics.update(memory_info)
        except Exception as e:
            print(f"Error getting CPU/memory metrics: {e}")
        
        # Get GPU metrics
        try:
            gpu_info = self.get_training_gpu_info()
            for gpu_id, info in gpu_info.items():
                for key, value in info.items():
                    if key != 'name' and isinstance(value, (int, float)):
                        metrics[f'{gpu_id}_{key}'] = value
        except Exception as e:
            print(f"Error getting GPU metrics: {e}")
        
        return metrics
    
    def monitor_loop(self):
        """Main monitoring loop focused on training process"""
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.monitoring:
            try:
                metrics = self.get_all_metrics()
                
                if self.log_to_wandb and WANDB_AVAILABLE:
                    try:
                        # Log to wandb with training/ prefix for training-specific metrics
                        prefix = "training/" if self.monitor_process_only else "hardware/"
                        wandb_metrics = {f"{prefix}{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
                        wandb.log(wandb_metrics)
                        consecutive_errors = 0
                    except Exception as e:
                        consecutive_errors += 1
                        print(f"Error logging to wandb: {e}")
                        if consecutive_errors >= max_consecutive_errors:
                            print("Too many consecutive wandb errors, disabling wandb logging")
                            self.log_to_wandb = False
                
                # Print summary every 30 iterations
                if hasattr(self, '_monitor_count'):
                    self._monitor_count += 1
                else:
                    self._monitor_count = 1
                
                if self._monitor_count % 30 == 0:
                    self._print_training_summary(metrics)
                
            except Exception as e:
                consecutive_errors += 1
                print(f"Error in monitoring loop: {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    print("Too many consecutive errors, stopping monitoring")
                    break
            
            time.sleep(self.monitor_interval)
    
    def _print_training_summary(self, metrics):
        """Print training-specific resource summary"""
        print(f"\n--- Training Resource Summary ({time.strftime('%H:%M:%S')}) ---")
        
        if self.monitor_process_only:
            # Process-specific metrics
            process_cpu = metrics.get('process_cpu_percent', 0)
            process_memory = metrics.get('process_memory_gb', 0)
            training_memory = metrics.get('training_memory_gb', 0)
            peak_memory = metrics.get('peak_process_memory_gb', 0)
            memory_percent = metrics.get('process_memory_percent', 0)
            
            print(f"Training Process CPU: {process_cpu:.1f}%")
            print(f"Training Process Memory: {memory_percent:.1f}% ({process_memory:.2f}GB of system)")
            print(f"Memory for Training: {training_memory:.2f}GB (above baseline)")
            print(f"Peak Process Memory: {peak_memory:.2f}GB")
            
            # System context
            system_cpu = metrics.get('system_cpu_percent', 0)
            system_available = metrics.get('system_memory_available_gb', 0)
            print(f"System CPU: {system_cpu:.1f}%, Available RAM: {system_available:.1f}GB")
        else:
            # System-wide metrics (fallback)
            cpu_percent = metrics.get('cpu_percent', 0)
            memory_percent = metrics.get('memory_percent', 0)
            memory_used = metrics.get('memory_used_gb', 0)
            memory_total = metrics.get('memory_total_gb', 0)
            
            print(f"System CPU: {cpu_percent:.1f}%")
            print(f"System Memory: {memory_percent:.1f}% ({memory_used:.1f}GB/{memory_total:.1f}GB)")
        
        # GPU information
        if self.gpu_available:
            for i in range(self.gpu_count):
                gpu_util = metrics.get(f'gpu_{i}_utilization_percent', 0)
                gpu_allocated = metrics.get(f'gpu_{i}_allocated_gb', 0)
                gpu_total = metrics.get(f'gpu_{i}_total_memory_gb', 0)
                gpu_percent = metrics.get(f'gpu_{i}_memory_percent', 0)
                training_gpu = metrics.get(f'gpu_{i}_training_memory_gb', 0)
                peak_gpu = metrics.get(f'gpu_{i}_peak_allocated_gb', 0)
                gpu_temp = metrics.get(f'gpu_{i}_temperature_c', 0)
                
                print(f"GPU {i}: {gpu_percent:.1f}% VRAM ({gpu_allocated:.2f}GB/{gpu_total:.1f}GB)")
                print(f"  Training GPU Memory: {training_gpu:.2f}GB (above baseline)")
                print(f"  Peak GPU Memory: {peak_gpu:.2f}GB")
                if gpu_util > 0:
                    print(f"  GPU Utilization: {gpu_util:.1f}%")
                if gpu_temp > 0:
                    print(f"  Temperature: {gpu_temp}°C")
        
        print("------------------------\n")
    
    def reset_baselines(self):
        """Reset baseline measurements (call this when training starts)"""
        try:
            self.baseline_memory = self.current_process.memory_info().rss / (1024**3)
            print(f"Reset baseline process memory: {self.baseline_memory:.2f}GB")
        except Exception:
            pass
        
        if self.gpu_available:
            self.baseline_gpu_memory = {i: torch.cuda.memory_allocated(i) / (1024**3) 
                                      for i in range(self.gpu_count)}
            print(f"Reset baseline GPU memory: {dict(self.baseline_gpu_memory)}")
        
        # Reset peaks
        self.peak_process_memory = self.baseline_memory
        self.peak_gpu_memory = self.baseline_gpu_memory.copy() if self.gpu_available else 0
    
    def get_training_summary(self):
        """Get final training resource summary"""
        try:
            current_memory = self.current_process.memory_info().rss / (1024**3)
            training_memory_used = current_memory - self.baseline_memory
            
            summary = {
                'baseline_memory_gb': self.baseline_memory,
                'final_memory_gb': current_memory,
                'training_memory_used_gb': training_memory_used,
                'peak_memory_gb': self.peak_process_memory
            }
            
            if self.gpu_available:
                for i in range(self.gpu_count):
                    baseline_gpu = self.baseline_gpu_memory.get(i, 0)
                    current_gpu = torch.cuda.memory_allocated(i) / (1024**3)
                    peak_gpu = self.peak_gpu_memory.get(i, 0) if isinstance(self.peak_gpu_memory, dict) else current_gpu
                    
                    summary[f'gpu_{i}_baseline_gb'] = baseline_gpu
                    summary[f'gpu_{i}_final_gb'] = current_gpu
                    summary[f'gpu_{i}_training_used_gb'] = current_gpu - baseline_gpu
                    summary[f'gpu_{i}_peak_gb'] = peak_gpu
            
            return summary
        except Exception as e:
            print(f"Error getting training summary: {e}")
            return {}
    
    def start(self):
        """Start monitoring"""
        if self.monitoring:
            print("Monitoring already started")
            return
        
        print("Starting training-specific resource monitoring...")
        self.monitoring = True
        self.thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.thread.start()
        print("Training resource monitoring started")
    
    def stop(self):
        """Stop monitoring and print final summary"""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=2.0)
        
        # Print final summary
        final_summary = self.get_training_summary()
        print(f"\n=== Final Training Resource Summary ===")
        print(f"Process Memory: {final_summary.get('baseline_memory_gb', 0):.2f}GB → {final_summary.get('final_memory_gb', 0):.2f}GB")
        print(f"Training Used: {final_summary.get('training_memory_used_gb', 0):.2f}GB (Peak: {final_summary.get('peak_memory_gb', 0):.2f}GB)")
        
        if self.gpu_available:
            for i in range(self.gpu_count):
                baseline = final_summary.get(f'gpu_{i}_baseline_gb', 0)
                final = final_summary.get(f'gpu_{i}_final_gb', 0)
                used = final_summary.get(f'gpu_{i}_training_used_gb', 0)
                peak = final_summary.get(f'gpu_{i}_peak_gb', 0)
                print(f"GPU {i}: {baseline:.2f}GB → {final:.2f}GB (Used: {used:.2f}GB, Peak: {peak:.2f}GB)")
        
        print("=======================================")
        print("Training resource monitoring stopped")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
