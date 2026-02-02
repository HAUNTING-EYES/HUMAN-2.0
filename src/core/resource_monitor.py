import psutil
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from collections import deque

@dataclass
class ResourceMetrics:
    cpu_percent: float
    memory_percent: float
    memory_used: int
    memory_available: int
    disk_usage_percent: float
    disk_io: Tuple[int, int]  # (read_bytes, write_bytes)
    network_io: Tuple[int, int]  # (bytes_sent, bytes_recv)
    timestamp: float

class ResourceMonitor:
    def __init__(self, history_size: int = 1000):
        """Initialize resource monitor with history tracking"""
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        self.baseline_metrics: Optional[ResourceMetrics] = None
        self.anomaly_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_usage_percent': 90.0
        }
        
        # Initialize baseline
        self._establish_baseline()

    def _establish_baseline(self, samples: int = 60):
        """Establish baseline metrics over multiple samples"""
        baseline_samples = []
        for _ in range(samples):
            metrics = self._collect_current_metrics()
            baseline_samples.append(metrics)
            time.sleep(1)  # 1-second intervals
            
        # Calculate average baseline
        self.baseline_metrics = ResourceMetrics(
            cpu_percent=np.mean([m.cpu_percent for m in baseline_samples]),
            memory_percent=np.mean([m.memory_percent for m in baseline_samples]),
            memory_used=np.mean([m.memory_used for m in baseline_samples]),
            memory_available=np.mean([m.memory_available for m in baseline_samples]),
            disk_usage_percent=np.mean([m.disk_usage_percent for m in baseline_samples]),
            disk_io=(
                np.mean([m.disk_io[0] for m in baseline_samples]),
                np.mean([m.disk_io[1] for m in baseline_samples])
            ),
            network_io=(
                np.mean([m.network_io[0] for m in baseline_samples]),
                np.mean([m.network_io[1] for m in baseline_samples])
            ),
            timestamp=time.time()
        )

    def _collect_current_metrics(self) -> ResourceMetrics:
        """Collect current system resource metrics"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()

        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used=memory.used,
            memory_available=memory.available,
            disk_usage_percent=disk.percent,
            disk_io=(disk_io.read_bytes, disk_io.write_bytes),
            network_io=(network_io.bytes_sent, network_io.bytes_recv),
            timestamp=time.time()
        )

    def update(self) -> Dict[str, float]:
        """Update and analyze current resource usage"""
        current_metrics = self._collect_current_metrics()
        self.metrics_history.append(current_metrics)
        
        # Calculate relative load compared to baseline
        relative_load = self._calculate_relative_load(current_metrics)
        
        # Detect anomalies
        anomalies = self._detect_anomalies(current_metrics)
        
        # Calculate trends
        trends = self._analyze_trends()
        
        return {
            'current_load': relative_load,
            'anomalies': anomalies,
            'trends': trends
        }

    def _calculate_relative_load(self, current: ResourceMetrics) -> Dict[str, float]:
        """Calculate load relative to baseline"""
        if not self.baseline_metrics:
            return {}
            
        return {
            'cpu_load': current.cpu_percent / max(self.baseline_metrics.cpu_percent, 1),
            'memory_load': current.memory_percent / max(self.baseline_metrics.memory_percent, 1),
            'disk_load': current.disk_usage_percent / max(self.baseline_metrics.disk_usage_percent, 1),
            'io_load': (
                (current.disk_io[0] + current.disk_io[1]) /
                max(sum(self.baseline_metrics.disk_io), 1)
            ),
            'network_load': (
                (current.network_io[0] + current.network_io[1]) /
                max(sum(self.baseline_metrics.network_io), 1)
            )
        }

    def _detect_anomalies(self, metrics: ResourceMetrics) -> List[str]:
        """Detect resource usage anomalies"""
        anomalies = []
        
        if metrics.cpu_percent > self.anomaly_thresholds['cpu_percent']:
            anomalies.append(f'High CPU usage: {metrics.cpu_percent}%')
            
        if metrics.memory_percent > self.anomaly_thresholds['memory_percent']:
            anomalies.append(f'High memory usage: {metrics.memory_percent}%')
            
        if metrics.disk_usage_percent > self.anomaly_thresholds['disk_usage_percent']:
            anomalies.append(f'High disk usage: {metrics.disk_usage_percent}%')
            
        return anomalies

    def _analyze_trends(self) -> Dict[str, float]:
        """Analyze resource usage trends"""
        if len(self.metrics_history) < 2:
            return {}
            
        # Calculate trends over last 10 samples or all if less
        window = min(10, len(self.metrics_history))
        recent_metrics = list(self.metrics_history)[-window:]
        
        cpu_trend = np.polyfit(
            range(window),
            [m.cpu_percent for m in recent_metrics],
            1
        )[0]
        
        memory_trend = np.polyfit(
            range(window),
            [m.memory_percent for m in recent_metrics],
            1
        )[0]
        
        disk_trend = np.polyfit(
            range(window),
            [m.disk_usage_percent for m in recent_metrics],
            1
        )[0]
        
        return {
            'cpu_trend': cpu_trend,
            'memory_trend': memory_trend,
            'disk_trend': disk_trend
        }

    def get_resource_forecast(self, minutes_ahead: int = 5) -> Dict[str, float]:
        """Forecast resource usage based on current trends"""
        if len(self.metrics_history) < 10:
            return {}
            
        trends = self._analyze_trends()
        current = self.metrics_history[-1]
        
        return {
            'cpu_forecast': current.cpu_percent + (trends['cpu_trend'] * minutes_ahead * 60),
            'memory_forecast': current.memory_percent + (trends['memory_trend'] * minutes_ahead * 60),
            'disk_forecast': current.disk_usage_percent + (trends['disk_trend'] * minutes_ahead * 60)
        }

    def set_anomaly_threshold(self, metric: str, threshold: float):
        """Update anomaly detection threshold for a metric"""
        if metric in self.anomaly_thresholds:
            self.anomaly_thresholds[metric] = threshold 