import psutil
import time
import threading
import torch
import platform
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
    """Simplified hardware resource monitor for training"""
    
    def __init__(self, monitor_interval=5.0, log_to_wandb=True):
        """
        Initialize resource monitor
        
        Args:
            monitor_interval: How often to collect metrics (seconds)
            log_to_wandb: Whether to log metrics to wandb
        """
        self.monitor_interval = monitor_interval
        self.log_to_wandb = log_to_wandb and WANDB_AVAILABLE
        self.monitoring = False
        self.thread = None
        
        # GPU availability
        self.gpu_available = torch.cuda.is_available()
        self.nvml_available = NVML_AVAILABLE and self.gpu_available
        
        # Get current process
        self.current_process = psutil.Process(os.getpid())
        
        # Peak tracking
        self.peak_process_memory = 0
        self.peak_gpu_memory = {}
        
        # GPU setup
        if self.gpu_available:
            self.gpu_count = torch.cuda.device_count()
            if self.nvml_available:
                print(f"GPU monitoring enabled for {self.gpu_count} GPU(s) with NVML")
                # Initialize peak GPU memory
                for i in range(self.gpu_count):
                    self.peak_gpu_memory[i] = 0
            else:
                print(f"GPU detected but NVML not available. GPU monitoring disabled.")
                self.gpu_available = False
                self.gpu_count = 0
        else:
            self.gpu_count = 0
            print("GPU monitoring disabled (no CUDA available)")
        
        print(f"System platform: {platform.system()}")
        print(f"CPU cores: {psutil.cpu_count()}")
        
        if not WANDB_AVAILABLE and log_to_wandb:
            print("Warning: wandb logging requested but not available")
    
    def get_cpu_metrics(self):
        """Get CPU usage metrics"""
        try:
            # Training process CPU (can exceed 100% on multi-core)
            process_cpu = self.current_process.cpu_percent(interval=None)
            
            # System-wide CPU average
            system_cpu = psutil.cpu_percent(interval=None)
            
            # Calculate cores used by training process
            cpu_count = psutil.cpu_count()
            cores_used = process_cpu / 100.0  # Convert percentage to cores
            
            return {
                'training_process_cpu_percent': process_cpu,
                'system_cpu_percent': system_cpu,
                'training_cpu_cores_used': cores_used,
                'total_cpu_cores': cpu_count
            }
        except Exception as e:
            print(f"Error getting CPU metrics: {e}")
            return {
                'training_process_cpu_percent': 0,
                'system_cpu_percent': 0,
                'training_cpu_cores_used': 0,
                'total_cpu_cores': psutil.cpu_count()
            }
    
    def get_memory_metrics(self):
        """Get RAM usage metrics"""
        try:
            # Process memory
            memory_info = self.current_process.memory_info()
            process_memory_gb = memory_info.rss / (1024**3)  # Resident Set Size
            
            # Update peak
            self.peak_process_memory = max(self.peak_process_memory, process_memory_gb)
            
            # System memory
            system_memory = psutil.virtual_memory()
            system_memory_gb = system_memory.total / (1024**3)
            
            # Calculate percentage of system memory
            memory_percent = (process_memory_gb / system_memory_gb * 100) if system_memory_gb > 0 else 0
            
            return {
                'process_memory_gb': process_memory_gb,
                'peak_memory_gb': self.peak_process_memory,
                'memory_percent_of_system': memory_percent,
                'system_total_memory_gb': system_memory_gb
            }
        except Exception as e:
            print(f"Error getting memory metrics: {e}")
            return {
                'process_memory_gb': 0,
                'peak_memory_gb': 0,
                'memory_percent_of_system': 0,
                'system_total_memory_gb': 0
            }
    
    def get_gpu_metrics(self):
        """Get GPU metrics using NVML only"""
        if not self.gpu_available or not self.nvml_available:
            return {}
        
        gpu_metrics = {}
        try:
            for i in range(self.gpu_count):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                
                # Memory info
                memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                vram_used_gb = memory_info.used / (1024**3)
                vram_total_gb = memory_info.total / (1024**3)
                vram_percent = (vram_used_gb / vram_total_gb * 100) if vram_total_gb > 0 else 0
                
                # Update peak
                self.peak_gpu_memory[i] = max(self.peak_gpu_memory[i], vram_used_gb)
                
                # GPU utilization
                utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util_percent = utilization.gpu
                
                # Temperature
                temperature = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                
                # Store metrics for this GPU
                gpu_metrics[f'gpu_{i}_vram_used_gb'] = vram_used_gb
                gpu_metrics[f'gpu_{i}_vram_total_gb'] = vram_total_gb
                gpu_metrics[f'gpu_{i}_vram_percent'] = vram_percent
                gpu_metrics[f'gpu_{i}_peak_memory_gb'] = self.peak_gpu_memory[i]
                gpu_metrics[f'gpu_{i}_utilization_percent'] = gpu_util_percent
                gpu_metrics[f'gpu_{i}_temperature_c'] = temperature
                
        except Exception as e:
            print(f"Error getting GPU metrics: {e}")
        
        return gpu_metrics
    
    def get_all_metrics(self):
        """Collect all metrics"""
        metrics = {}
        
        # CPU metrics
        metrics.update(self.get_cpu_metrics())
        
        # Memory metrics
        metrics.update(self.get_memory_metrics())
        
        # GPU metrics
        metrics.update(self.get_gpu_metrics())
        
        return metrics
    
    def monitor_loop(self):
        """Main monitoring loop"""
        consecutive_errors = 0
        max_consecutive_errors = 5
        monitor_count = 0
        
        while self.monitoring:
            try:
                metrics = self.get_all_metrics()
                
                # Log to wandb
                if self.log_to_wandb and WANDB_AVAILABLE:
                    try:
                        wandb_metrics = {f"hw/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
                        wandb.log(wandb_metrics)
                        consecutive_errors = 0
                    except Exception as e:
                        consecutive_errors += 1
                        print(f"Error logging to wandb: {e}")
                        if consecutive_errors >= max_consecutive_errors:
                            print("Too many consecutive wandb errors, disabling wandb logging")
                            self.log_to_wandb = False
                
                # Print summary every 30 iterations
                monitor_count += 1
                if monitor_count % 30 == 0:
                    self._print_summary(metrics)
                
            except Exception as e:
                consecutive_errors += 1
                print(f"Error in monitoring loop: {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    print("Too many consecutive errors, stopping monitoring")
                    break
            
            time.sleep(self.monitor_interval)
    
    def _print_summary(self, metrics):
        """Print resource summary"""
        print(f"\n{'='*60}")
        print(f"Hardware Resource Summary - {time.strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        
        # CPU metrics
        process_cpu = metrics.get('training_process_cpu_percent', 0)
        system_cpu = metrics.get('system_cpu_percent', 0)
        cores_used = metrics.get('training_cpu_cores_used', 0)
        total_cores = metrics.get('total_cpu_cores', 0)
        
        print(f"\n📊 CPU Usage:")
        print(f"  Training Process: {process_cpu:.1f}% ({cores_used:.2f} cores)")
        print(f"  System-wide:      {system_cpu:.1f}%")
        print(f"  Total Cores:      {total_cores}")
        
        # Memory metrics
        process_mem = metrics.get('process_memory_gb', 0)
        peak_mem = metrics.get('peak_memory_gb', 0)
        mem_percent = metrics.get('memory_percent_of_system', 0)
        total_mem = metrics.get('system_total_memory_gb', 0)
        
        print(f"\n💾 RAM Usage:")
        print(f"  Process Memory:   {process_mem:.2f}GB ({mem_percent:.1f}% of {total_mem:.1f}GB)")
        print(f"  Peak Memory:      {peak_mem:.2f}GB")
        
        # GPU metrics
        if self.gpu_available:
            print(f"\n🎮 GPU Usage:")
            for i in range(self.gpu_count):
                vram_used = metrics.get(f'gpu_{i}_vram_used_gb', 0)
                vram_total = metrics.get(f'gpu_{i}_vram_total_gb', 0)
                vram_percent = metrics.get(f'gpu_{i}_vram_percent', 0)
                peak_gpu = metrics.get(f'gpu_{i}_peak_memory_gb', 0)
                gpu_util = metrics.get(f'gpu_{i}_utilization_percent', 0)
                gpu_temp = metrics.get(f'gpu_{i}_temperature_c', 0)
                
                print(f"  GPU {i}:")
                print(f"    VRAM:         {vram_used:.2f}GB / {vram_total:.2f}GB ({vram_percent:.1f}%)")
                print(f"    Peak VRAM:    {peak_gpu:.2f}GB")
                print(f"    Utilization:  {gpu_util:.1f}%")
                print(f"    Temperature:  {gpu_temp}°C")
        
        print(f"{'='*60}\n")
    
    def start(self):
        """Start monitoring"""
        if self.monitoring:
            print("Monitoring already started")
            return
        
        print("\n🚀 Starting hardware resource monitoring...")
        self.monitoring = True
        self.thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.thread.start()
        print("✅ Hardware monitoring started\n")
    
    def stop(self):
        """Stop monitoring and print final summary"""
        if not self.monitoring:
            return
        
        print("\n🛑 Stopping hardware monitoring...")
        self.monitoring = False
        
        if self.thread:
            self.thread.join(timeout=2.0)
        
        # Get final metrics
        final_metrics = self.get_all_metrics()
        
        # Print final summary
        print(f"\n{'='*60}")
        print(f"FINAL HARDWARE RESOURCE SUMMARY")
        print(f"{'='*60}")
        
        # CPU
        cores_used = final_metrics.get('training_cpu_cores_used', 0)
        total_cores = final_metrics.get('total_cpu_cores', 0)
        print(f"\n📊 CPU:")
        print(f"  Average cores used: {cores_used:.2f} / {total_cores}")
        
        # Memory
        peak_mem = final_metrics.get('peak_memory_gb', 0)
        current_mem = final_metrics.get('process_memory_gb', 0)
        print(f"\n💾 RAM:")
        print(f"  Current: {current_mem:.2f}GB")
        print(f"  Peak:    {peak_mem:.2f}GB")
        
        # GPU
        if self.gpu_available:
            print(f"\n🎮 GPU:")
            for i in range(self.gpu_count):
                peak_gpu = final_metrics.get(f'gpu_{i}_peak_memory_gb', 0)
                current_gpu = final_metrics.get(f'gpu_{i}_vram_used_gb', 0)
                max_temp = final_metrics.get(f'gpu_{i}_temperature_c', 0)
                
                print(f"  GPU {i}:")
                print(f"    Current VRAM: {current_gpu:.2f}GB")
                print(f"    Peak VRAM:    {peak_gpu:.2f}GB")
                print(f"    Temperature:  {max_temp}°C")
        
        print(f"{'='*60}\n")
        print("✅ Hardware monitoring stopped\n")
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


# Convenience function
def monitor_training_resources(monitor_interval=5.0, log_to_wandb=True):
    """
    Create a resource monitor for training
    
    Args:
        monitor_interval: How often to check resources (seconds)
        log_to_wandb: Whether to log to wandb
    
    Returns:
        ResourceMonitor instance
    
    Example:
        with monitor_training_resources(monitor_interval=5.0) as monitor:
            train_model()
    """
    return ResourceMonitor(
        monitor_interval=monitor_interval,
        log_to_wandb=log_to_wandb
    )


# Test function
if __name__ == "__main__":
    print("Testing Hardware Resource Monitor...\n")
    
    with monitor_training_resources(monitor_interval=2.0, log_to_wandb=False) as monitor:
        print("Simulating training for 10 seconds...\n")
        
        # Simulate some work
        if torch.cuda.is_available():
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.randn(1000, 1000, device='cuda')
            for _ in range(5):
                z = torch.matmul(x, y)
                time.sleep(1)
        else:
            # CPU work
            import numpy as np
            for _ in range(5):
                data = np.random.randn(5000, 5000)
                result = np.dot(data, data.T)
                time.sleep(1)
    
    print("\n✅ Test completed!")