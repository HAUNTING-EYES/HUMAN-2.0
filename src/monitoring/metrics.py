import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, start_http_server

@dataclass
class OptimizationMetrics:
    """Metrics for a single code optimization."""
    request_id: str
    original_code_length: int
    optimized_code_length: int
    optimization_time: float
    num_steps: int
    improvements: List[str]
    metrics: Dict[str, float]
    timestamp: str

class MetricsCollector:
    """Collector for code optimization metrics."""
    
    def __init__(self, metrics_dir: str, enable_prometheus: bool = True):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Initialize Prometheus metrics
        if enable_prometheus:
            # Request metrics
            self.request_counter = Counter(
                'code_optimization_requests_total',
                'Total number of code optimization requests'
            )
            self.request_latency = Histogram(
                'code_optimization_request_latency_seconds',
                'Latency of code optimization requests',
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            )
            
            # Optimization metrics
            self.optimization_steps = Histogram(
                'code_optimization_steps',
                'Number of optimization steps taken',
                buckets=[1, 2, 5, 10, 20]
            )
            self.code_length_ratio = Histogram(
                'code_optimization_length_ratio',
                'Ratio of optimized code length to original code length',
                buckets=[0.5, 0.75, 1.0, 1.25, 1.5]
            )
            
            # Quality metrics
            self.complexity_gauge = Gauge(
                'code_optimization_complexity',
                'Code complexity metric'
            )
            self.maintainability_gauge = Gauge(
                'code_optimization_maintainability',
                'Code maintainability metric'
            )
            self.performance_gauge = Gauge(
                'code_optimization_performance',
                'Code performance metric'
            )
            
            # Start Prometheus server
            start_http_server(8001)
            self.logger.info("Prometheus metrics server started on port 8001")
            
    def record_optimization(
        self,
        request_id: str,
        original_code: str,
        optimized_code: str,
        optimization_time: float,
        num_steps: int,
        improvements: List[str],
        metrics: Dict[str, float]
    ) -> OptimizationMetrics:
        """Record metrics for a code optimization."""
        # Create metrics object
        optimization_metrics = OptimizationMetrics(
            request_id=request_id,
            original_code_length=len(original_code),
            optimized_code_length=len(optimized_code),
            optimization_time=optimization_time,
            num_steps=num_steps,
            improvements=improvements,
            metrics=metrics,
            timestamp=datetime.now().isoformat()
        )
        
        # Save metrics to file
        self._save_metrics(optimization_metrics)
        
        # Update Prometheus metrics
        self._update_prometheus_metrics(optimization_metrics)
        
        return optimization_metrics
        
    def _save_metrics(self, metrics: OptimizationMetrics):
        """Save metrics to file."""
        metrics_file = self.metrics_dir / f"{metrics.request_id}.json"
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(metrics.__dict__, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")
            
    def _update_prometheus_metrics(self, metrics: OptimizationMetrics):
        """Update Prometheus metrics."""
        try:
            # Update request metrics
            self.request_counter.inc()
            self.request_latency.observe(metrics.optimization_time)
            
            # Update optimization metrics
            self.optimization_steps.observe(metrics.num_steps)
            length_ratio = metrics.optimized_code_length / metrics.original_code_length
            self.code_length_ratio.observe(length_ratio)
            
            # Update quality metrics
            self.complexity_gauge.set(metrics.metrics.get('complexity', 0))
            self.maintainability_gauge.set(metrics.metrics.get('maintainability', 0))
            self.performance_gauge.set(metrics.metrics.get('performance', 0))
            
        except Exception as e:
            self.logger.error(f"Error updating Prometheus metrics: {str(e)}")
            
    def get_metrics_summary(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> Dict[str, float]:
        """Get summary of metrics within time range."""
        metrics_list = []
        
        # Load all metrics files
        for metrics_file in self.metrics_dir.glob("*.json"):
            try:
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                    
                # Filter by time range if specified
                if start_time and metrics_data['timestamp'] < start_time:
                    continue
                if end_time and metrics_data['timestamp'] > end_time:
                    continue
                    
                metrics_list.append(metrics_data)
                
            except Exception as e:
                self.logger.error(f"Error loading metrics file {metrics_file}: {str(e)}")
                
        if not metrics_list:
            return {}
            
        # Calculate summary statistics
        summary = {
            'total_requests': len(metrics_list),
            'avg_optimization_time': float(np.mean([m['optimization_time'] for m in metrics_list])),
            'avg_num_steps': float(np.mean([m['num_steps'] for m in metrics_list])),
            'avg_code_length_ratio': float(np.mean([
                m['optimized_code_length'] / m['original_code_length']
                for m in metrics_list
            ])),
            'avg_complexity': float(np.mean([
                m['metrics'].get('complexity', 0)
                for m in metrics_list
            ])),
            'avg_maintainability': float(np.mean([
                m['metrics'].get('maintainability', 0)
                for m in metrics_list
            ])),
            'avg_performance': float(np.mean([
                m['metrics'].get('performance', 0)
                for m in metrics_list
            ]))
        }
        
        return summary
        
    def get_improvement_stats(self) -> Dict[str, int]:
        """Get statistics about types of improvements made."""
        improvement_counts = {}
        
        # Load all metrics files
        for metrics_file in self.metrics_dir.glob("*.json"):
            try:
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                    
                # Count improvements
                for improvement in metrics_data['improvements']:
                    improvement_type = improvement.split(":")[1].strip()
                    improvement_counts[improvement_type] = (
                        improvement_counts.get(improvement_type, 0) + 1
                    )
                    
            except Exception as e:
                self.logger.error(f"Error loading metrics file {metrics_file}: {str(e)}")
                
        return improvement_counts 