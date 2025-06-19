"""
Health monitoring and system diagnostics for the SCoRe system.
"""
import time
import threading
import logging
import psutil
import torch
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    status: HealthStatus
    value: Any
    threshold: Optional[float] = None
    message: str = ""
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

@dataclass
class SystemHealthReport:
    """Complete system health report."""
    overall_status: HealthStatus
    timestamp: float
    metrics: List[HealthMetric]
    system_info: Dict[str, Any]
    errors: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'overall_status': self.overall_status.value,
            'timestamp': self.timestamp,
            'metrics': [asdict(metric) for metric in self.metrics],
            'system_info': self.system_info,
            'errors': self.errors
        }

class HealthChecker:
    """Comprehensive health monitoring system."""
    
    def __init__(self, config, model=None, trainer=None):
        self.config = config
        self.model = model
        self.trainer = trainer
        self.checks: Dict[str, Callable] = {}
        self.thresholds = {
            'cpu_percent': 90.0,
            'memory_percent': 85.0,
            'gpu_memory_percent': 90.0,
            'disk_usage_percent': 95.0,
            'temperature': 80.0,
        }
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        self.health_history: List[SystemHealthReport] = []
        self.max_history_size = 100
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks."""
        self.checks.update({
            'system_resources': self._check_system_resources,
            'disk_space': self._check_disk_space,
            'model_health': self._check_model_health,
            'trainer_health': self._check_trainer_health,
            'memory_usage': self._check_memory_usage,
            'gpu_health': self._check_gpu_health,
            'configuration': self._check_configuration,
        })
    
    def register_check(self, name: str, check_func: Callable) -> None:
        """Register a custom health check."""
        self.checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    def _check_system_resources(self) -> List[HealthMetric]:
        """Check system CPU and memory usage."""
        metrics = []
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_status = HealthStatus.HEALTHY
            if cpu_percent > self.thresholds['cpu_percent']:
                cpu_status = HealthStatus.CRITICAL
            elif cpu_percent > self.thresholds['cpu_percent'] * 0.8:
                cpu_status = HealthStatus.WARNING
            
            metrics.append(HealthMetric(
                name="cpu_usage",
                status=cpu_status,
                value=cpu_percent,
                threshold=self.thresholds['cpu_percent'],
                message=f"CPU usage at {cpu_percent:.1f}%"
            ))
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_status = HealthStatus.HEALTHY
            if memory.percent > self.thresholds['memory_percent']:
                memory_status = HealthStatus.CRITICAL
            elif memory.percent > self.thresholds['memory_percent'] * 0.8:
                memory_status = HealthStatus.WARNING
            
            metrics.append(HealthMetric(
                name="memory_usage",
                status=memory_status,
                value=memory.percent,
                threshold=self.thresholds['memory_percent'],
                message=f"Memory usage at {memory.percent:.1f}%"
            ))
            
        except Exception as e:
            metrics.append(HealthMetric(
                name="system_resources_error",
                status=HealthStatus.CRITICAL,
                value=str(e),
                message=f"Failed to check system resources: {e}"
            ))
        
        return metrics
    
    def _check_disk_space(self) -> List[HealthMetric]:
        """Check disk space usage."""
        metrics = []
        
        try:
            paths_to_check = [
                self.config.data_path,
                self.config.output_dir,
                self.config.log_dir
            ]
            
            for path in paths_to_check:
                if Path(path).exists():
                    disk_usage = psutil.disk_usage(path)
                    usage_percent = (disk_usage.used / disk_usage.total) * 100
                    
                    status = HealthStatus.HEALTHY
                    if usage_percent > self.thresholds['disk_usage_percent']:
                        status = HealthStatus.CRITICAL
                    elif usage_percent > self.thresholds['disk_usage_percent'] * 0.9:
                        status = HealthStatus.WARNING
                    
                    metrics.append(HealthMetric(
                        name=f"disk_usage_{Path(path).name}",
                        status=status,
                        value=usage_percent,
                        threshold=self.thresholds['disk_usage_percent'],
                        message=f"Disk usage for {path}: {usage_percent:.1f}%"
                    ))
                    
        except Exception as e:
            metrics.append(HealthMetric(
                name="disk_space_error",
                status=HealthStatus.CRITICAL,
                value=str(e),
                message=f"Failed to check disk space: {e}"
            ))
        
        return metrics
    
    def _check_model_health(self) -> List[HealthMetric]:
        """Check model health status."""
        metrics = []
        
        if self.model is None:
            metrics.append(HealthMetric(
                name="model_not_available",
                status=HealthStatus.WARNING,
                value=False,
                message="Model not available for health check"
            ))
            return metrics
        
        try:
            health = self.model.health_check()
            
            status = HealthStatus.HEALTHY if health.is_healthy else HealthStatus.CRITICAL
            metrics.append(HealthMetric(
                name="model_health",
                status=status,
                value=health.is_healthy,
                message=health.last_error or "Model health check passed"
            ))
            
            # Model error count
            error_status = HealthStatus.HEALTHY
            if health.error_count > 10:
                error_status = HealthStatus.CRITICAL
            elif health.error_count > 5:
                error_status = HealthStatus.WARNING
            
            metrics.append(HealthMetric(
                name="model_error_count",
                status=error_status,
                value=health.error_count,
                threshold=10,
                message=f"Model has {health.error_count} recorded errors"
            ))
            
        except Exception as e:
            metrics.append(HealthMetric(
                name="model_health_error",
                status=HealthStatus.CRITICAL,
                value=str(e),
                message=f"Failed to check model health: {e}"
            ))
        
        return metrics
    
    def _check_trainer_health(self) -> List[HealthMetric]:
        """Check trainer health status."""
        metrics = []
        
        if self.trainer is None:
            metrics.append(HealthMetric(
                name="trainer_not_available",
                status=HealthStatus.WARNING,
                value=False,
                message="Trainer not available for health check"
            ))
            return metrics
        
        try:
            # Check if trainer is in a valid state
            if hasattr(self.trainer, 'global_step') and self.trainer.global_step >= 0:
                metrics.append(HealthMetric(
                    name="trainer_state",
                    status=HealthStatus.HEALTHY,
                    value=self.trainer.global_step,
                    message=f"Trainer at step {self.trainer.global_step}"
                ))
            else:
                metrics.append(HealthMetric(
                    name="trainer_state",
                    status=HealthStatus.WARNING,
                    value="unknown",
                    message="Trainer state unclear"
                ))
                
        except Exception as e:
            metrics.append(HealthMetric(
                name="trainer_health_error",
                status=HealthStatus.CRITICAL,
                value=str(e),
                message=f"Failed to check trainer health: {e}"
            ))
        
        return metrics
    
    def _check_memory_usage(self) -> List[HealthMetric]:
        """Check memory usage details."""
        metrics = []
        
        try:
            # System memory
            memory = psutil.virtual_memory()
            metrics.append(HealthMetric(
                name="available_memory_gb",
                status=HealthStatus.HEALTHY,
                value=memory.available / (1024**3),
                message=f"Available memory: {memory.available / (1024**3):.1f} GB"
            ))
            
            # GPU memory if available
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    
                    metrics.append(HealthMetric(
                        name=f"gpu_{i}_memory_allocated_gb",
                        status=HealthStatus.HEALTHY,
                        value=allocated,
                        message=f"GPU {i} allocated: {allocated:.1f} GB"
                    ))
                    
                    metrics.append(HealthMetric(
                        name=f"gpu_{i}_memory_reserved_gb",
                        status=HealthStatus.HEALTHY,
                        value=reserved,
                        message=f"GPU {i} reserved: {reserved:.1f} GB"
                    ))
                    
        except Exception as e:
            metrics.append(HealthMetric(
                name="memory_usage_error",
                status=HealthStatus.CRITICAL,
                value=str(e),
                message=f"Failed to check memory usage: {e}"
            ))
        
        return metrics
    
    def _check_gpu_health(self) -> List[HealthMetric]:
        """Check GPU health and availability."""
        metrics = []
        
        try:
            if not torch.cuda.is_available():
                metrics.append(HealthMetric(
                    name="cuda_available",
                    status=HealthStatus.WARNING,
                    value=False,
                    message="CUDA not available"
                ))
                return metrics
            
            # Check each GPU
            for i in range(torch.cuda.device_count()):
                try:
                    # Test basic GPU operation
                    device = torch.device(f'cuda:{i}')
                    test_tensor = torch.randn(100, 100, device=device)
                    _ = test_tensor @ test_tensor.T
                    del test_tensor
                    
                    metrics.append(HealthMetric(
                        name=f"gpu_{i}_functional",
                        status=HealthStatus.HEALTHY,
                        value=True,
                        message=f"GPU {i} functional test passed"
                    ))
                    
                except Exception as e:
                    metrics.append(HealthMetric(
                        name=f"gpu_{i}_functional",
                        status=HealthStatus.CRITICAL,
                        value=False,
                        message=f"GPU {i} failed functional test: {e}"
                    ))
                    
        except Exception as e:
            metrics.append(HealthMetric(
                name="gpu_health_error",
                status=HealthStatus.CRITICAL,
                value=str(e),
                message=f"Failed to check GPU health: {e}"
            ))
        
        return metrics
    
    def _check_configuration(self) -> List[HealthMetric]:
        """Check configuration validity."""
        metrics = []
        
        try:
            # Validate config
            self.config.validate()
            metrics.append(HealthMetric(
                name="configuration_valid",
                status=HealthStatus.HEALTHY,
                value=True,
                message="Configuration validation passed"
            ))
            
            # Check critical paths
            critical_paths = {
                'data_path': self.config.data_path,
                'output_dir': self.config.output_dir,
                'log_dir': self.config.log_dir
            }
            
            for name, path in critical_paths.items():
                if Path(path).exists() and Path(path).is_dir():
                    metrics.append(HealthMetric(
                        name=f"{name}_accessible",
                        status=HealthStatus.HEALTHY,
                        value=True,
                        message=f"{name} is accessible at {path}"
                    ))
                else:
                    metrics.append(HealthMetric(
                        name=f"{name}_accessible",
                        status=HealthStatus.CRITICAL,
                        value=False,
                        message=f"{name} not accessible at {path}"
                    ))
                    
        except Exception as e:
            metrics.append(HealthMetric(
                name="configuration_error",
                status=HealthStatus.CRITICAL,
                value=str(e),
                message=f"Configuration validation failed: {e}"
            ))
        
        return metrics
    
    def run_all_checks(self) -> SystemHealthReport:
        """Run all registered health checks."""
        start_time = time.time()
        all_metrics = []
        errors = []
        
        # Get system info
        system_info = {
            'timestamp': start_time,
            'platform': psutil.platform,
            'cpu_count': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}",
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            system_info['gpu_count'] = torch.cuda.device_count()
            system_info['cuda_version'] = torch.version.cuda
        
        # Run all checks
        for check_name, check_func in self.checks.items():
            try:
                metrics = check_func()
                all_metrics.extend(metrics)
                logger.debug(f"Health check '{check_name}' completed with {len(metrics)} metrics")
            except Exception as e:
                error_msg = f"Health check '{check_name}' failed: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
                
                all_metrics.append(HealthMetric(
                    name=f"{check_name}_failed",
                    status=HealthStatus.CRITICAL,
                    value=str(e),
                    message=error_msg
                ))
        
        # Determine overall status
        overall_status = HealthStatus.HEALTHY
        for metric in all_metrics:
            if metric.status == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
                break
            elif metric.status == HealthStatus.WARNING and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.WARNING
        
        # Create health report
        report = SystemHealthReport(
            overall_status=overall_status,
            timestamp=start_time,
            metrics=all_metrics,
            system_info=system_info,
            errors=errors
        )
        
        # Store in history
        self.health_history.append(report)
        if len(self.health_history) > self.max_history_size:
            self.health_history.pop(0)
        
        logger.info(f"Health check completed in {time.time() - start_time:.2f}s with status: {overall_status.value}")
        return report
    
    def start_monitoring(self, interval: int = 300) -> None:
        """Start continuous health monitoring."""
        if self.monitoring_active:
            logger.warning("Health monitoring already active")
            return
        
        self.monitoring_active = True
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    report = self.run_all_checks()
                    
                    # Save report to file
                    if self.config.enable_health_checks:
                        report_file = Path(self.config.log_dir) / 'health_reports.jsonl'
                        with open(report_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(report.to_dict()) + '\n')
                    
                    # Log critical issues
                    if report.overall_status == HealthStatus.CRITICAL:
                        logger.error(f"CRITICAL health issues detected: {len(report.errors)} errors")
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"Error in health monitoring loop: {e}")
                    time.sleep(interval)
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info(f"Health monitoring started with {interval}s interval")
    
    def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        if not self.monitoring_active:
            logger.warning("Health monitoring not active")
            return
        
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Health monitoring stopped")
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of recent health status."""
        if not self.health_history:
            return {'status': 'no_data', 'message': 'No health data available'}
        
        latest_report = self.health_history[-1]
        
        # Count metrics by status
        status_counts = {status.value: 0 for status in HealthStatus}
        for metric in latest_report.metrics:
            status_counts[metric.status.value] += 1
        
        return {
            'overall_status': latest_report.overall_status.value,
            'timestamp': latest_report.timestamp,
            'metric_counts': status_counts,
            'error_count': len(latest_report.errors),
            'total_checks': len(latest_report.metrics),
            'uptime_hours': (time.time() - self.health_history[0].timestamp) / 3600 if len(self.health_history) > 1 else 0
        }
    
    def export_health_report(self, filepath: str) -> None:
        """Export detailed health report to file."""
        if not self.health_history:
            raise ValueError("No health data to export")
        
        report_data = {
            'export_timestamp': time.time(),
            'config_info': {
                'task': self.config.task,
                'model_variant': self.config.model_variant,
                'environment': self.config.environment
            },
            'health_reports': [report.to_dict() for report in self.health_history]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Health report exported to {filepath}")

def create_health_checker(config, model=None, trainer=None) -> HealthChecker:
    """Factory function to create a health checker."""
    return HealthChecker(config, model, trainer) 