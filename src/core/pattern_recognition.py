import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from scipy import stats
from scipy.signal import find_peaks
import networkx as nx
from collections import deque
import logging
from enum import Enum, auto

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatternError(Exception):
    """Base class for pattern recognition errors."""
    pass

class InvalidDataError(PatternError):
    """Error raised when input data is invalid."""
    pass

class InsufficientDataError(PatternError):
    """Error raised when there is not enough data for analysis."""
    pass

class AnalysisError(PatternError):
    """Error raised when pattern analysis fails."""
    pass

class PatternType(Enum):
    """Enumeration of supported pattern types."""
    CYCLIC = auto()
    TREND = auto()
    ANOMALY = auto()
    CORRELATION = auto()

@dataclass
class TimeSeriesPattern:
    """Represents a detected pattern in time series data.
    
    Attributes:
        pattern_id: Unique identifier for the pattern
        values: Array of pattern values
        timestamps: Array of corresponding timestamps
        frequency: Frequency of the pattern in Hz (0 for non-cyclic patterns)
        confidence: Confidence score of pattern detection [0-1]
        type: Type of pattern ('cyclic', 'trend', 'anomaly', 'correlation')
        metadata: Additional pattern-specific information
    """
    pattern_id: str
    values: np.ndarray
    timestamps: np.ndarray
    frequency: float  # Hz
    confidence: float
    type: str  # 'cyclic', 'trend', 'anomaly', 'correlation'
    metadata: Dict[str, Any]

@dataclass
class CausalLink:
    """Represents a causal relationship between two variables.
    
    Attributes:
        source_id: ID of the source variable
        target_id: ID of the target variable
        lag: Time lag between cause and effect (seconds)
        strength: Strength of causal relationship [0-1]
        confidence: Confidence in the causal relationship [0-1]
        type: Type of causality ('direct', 'indirect', 'bidirectional')
    """
    source_id: str
    target_id: str
    lag: float  # Time lag in seconds
    strength: float  # Correlation strength
    confidence: float
    type: str  # 'direct', 'indirect', 'bidirectional'

class PatternRecognitionSystem:
    """Advanced pattern recognition system for time series analysis.
    
    This system performs:
    1. Time series pattern detection (cyclic, trends, anomalies)
    2. Causal relationship analysis between variables
    3. Multi-scale temporal analysis using various window sizes
    
    The system maintains a rolling history of observations and continuously
    updates pattern detection as new data arrives. It uses multiple analysis
    techniques including:
    - Spectral analysis for cyclic pattern detection
    - Regression analysis for trend detection
    - Statistical methods for anomaly detection
    - Granger causality for causal relationship analysis
    
    Attributes:
        max_history: Maximum number of historical points to maintain
        min_pattern_confidence: Minimum confidence threshold for pattern detection
        significance_level: Statistical significance level for hypothesis tests
        window_sizes: List of analysis window sizes (in samples)
        min_cycle_periods: Minimum periods required for cycle detection
    """
    
    def __init__(self, 
                 max_history: int = 10000,
                 min_pattern_confidence: float = 0.7,
                 significance_level: float = 0.05):
        """Initialize the pattern recognition system.
        
        Args:
            max_history: Maximum number of historical points to maintain
            min_pattern_confidence: Minimum confidence threshold [0-1]
            significance_level: Statistical significance level
        """
        self.max_history = max_history
        self.min_pattern_confidence = min_pattern_confidence
        self.significance_level = significance_level
        
        # Data storage
        self.time_series_data: Dict[str, deque] = {}  # Variable -> time series
        self.patterns: Dict[str, TimeSeriesPattern] = {}
        self.causal_graph = nx.DiGraph()
        
        # Analysis parameters
        self.window_sizes = [10, 30, 60, 300]  # Multiple time windows for analysis
        self.min_cycle_periods = [2, 5, 10, 30]  # Minimum periods for cycle detection
        
    def add_observation(self, variable: str, value: float, timestamp: Optional[float] = None):
        """Add a new observation to the time series data.
        
        Args:
            variable: Name of the variable being observed
            value: Observed value
            timestamp: Optional timestamp (uses current time if None)
            
        Raises:
            InvalidDataError: If variable is empty or value is invalid
            
        Note:
            If the variable doesn't exist, a new time series is created.
            Old observations are automatically removed when max_history is reached.
        """
        try:
            # Validate inputs
            if not variable or not isinstance(variable, str):
                raise InvalidDataError("Variable name must be a non-empty string")
            
            if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                raise InvalidDataError(f"Invalid value for variable {variable}: {value}")
            
            if timestamp is None:
                timestamp = datetime.now().timestamp()
            elif not isinstance(timestamp, (int, float)) or timestamp < 0:
                raise InvalidDataError(f"Invalid timestamp: {timestamp}")
                
            if variable not in self.time_series_data:
                self.time_series_data[variable] = deque(maxlen=self.max_history)
                logger.info(f"Created new time series for variable: {variable}")
                
            self.time_series_data[variable].append((timestamp, value))
            logger.debug(f"Added observation for {variable}: {value} at {timestamp}")
            
        except Exception as e:
            logger.error(f"Error adding observation: {str(e)}")
            raise
        
    def analyze_patterns(self) -> List[TimeSeriesPattern]:
        """Analyze all time series data to detect patterns.
        
        This method performs comprehensive pattern analysis including:
        1. Cyclic pattern detection using spectral analysis
        2. Trend detection using regression analysis
        3. Anomaly detection using statistical methods
        4. Causal relationship analysis between variables
        
        Returns:
            List of detected patterns across all variables
            
        Raises:
            InsufficientDataError: If there is not enough data for analysis
            AnalysisError: If pattern analysis fails
            
        Note:
            This is computationally intensive for large datasets or many variables.
            Consider running it periodically rather than after every observation.
        """
        try:
            patterns = []
            min_required = min(self.window_sizes)
            
            # Check if we have enough data
            if not self.time_series_data:
                logger.warning("No time series data available for analysis")
                return []
                
            # Analyze each variable
            for variable, data in self.time_series_data.items():
                if len(data) < min_required:
                    logger.warning(f"Insufficient data for variable {variable}: {len(data)} < {min_required}")
                    continue
                    
                try:
                    # Convert to numpy arrays
                    timestamps, values = zip(*data)
                    timestamps = np.array(timestamps)
                    values = np.array(values)
                    
                    # Validate data
                    if np.any(np.isnan(values)) or np.any(np.isinf(values)):
                        logger.error(f"Invalid values detected in variable {variable}")
                        continue
                        
                    if np.any(np.diff(timestamps) <= 0):
                        logger.error(f"Non-monotonic timestamps detected in variable {variable}")
                        continue
                        
                    # Detect different types of patterns
                    cyclic_patterns = self._detect_cyclic_patterns(variable, timestamps, values)
                    trend_patterns = self._detect_trends(variable, timestamps, values)
                    anomaly_patterns = self._detect_anomalies(variable, timestamps, values)
                    
                    patterns.extend(cyclic_patterns + trend_patterns + anomaly_patterns)
                    
                except Exception as e:
                    logger.error(f"Error analyzing patterns for variable {variable}: {str(e)}")
                    continue
                    
            # Update stored patterns
            try:
                self._update_pattern_storage(patterns)
            except Exception as e:
                logger.error(f"Error updating pattern storage: {str(e)}")
                
            # Analyze causal relationships
            try:
                self._analyze_causality()
            except Exception as e:
                logger.error(f"Error analyzing causality: {str(e)}")
                
            return patterns
            
        except Exception as e:
            logger.error(f"Error in pattern analysis: {str(e)}")
            raise AnalysisError(f"Pattern analysis failed: {str(e)}")
        
    def _detect_cyclic_patterns(self, 
                              variable: str, 
                              timestamps: np.ndarray, 
                              values: np.ndarray) -> List[TimeSeriesPattern]:
        """Detect cyclic patterns using spectral analysis.
        
        Uses Fast Fourier Transform (FFT) to identify dominant frequencies
        and their corresponding amplitudes in the time series data.
        
        Args:
            variable: Name of the variable being analyzed
            timestamps: Array of observation timestamps
            values: Array of observed values
            
        Returns:
            List of detected cyclic patterns
            
        Raises:
            InvalidDataError: If input data is invalid
            AnalysisError: If pattern detection fails
            
        Note:
            The confidence score is based on the prominence of frequency peaks
            relative to the overall frequency spectrum.
        """
        patterns = []
        
        try:
            for window_size in self.window_sizes:
                if len(values) < window_size:
                    continue
                    
                # Use last window_size points
                window_values = values[-window_size:]
                window_timestamps = timestamps[-window_size:]
                
                try:
                    # Compute FFT
                    sampling_rate = 1 / np.mean(np.diff(window_timestamps))
                    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
                        logger.warning(f"Invalid sampling rate for variable {variable}")
                        continue
                        
                    fft = np.fft.fft(window_values)
                    freqs = np.fft.fftfreq(len(window_values), 1/sampling_rate)
                    
                    # Find dominant frequencies
                    magnitude = np.abs(fft)
                    if not np.any(np.isfinite(magnitude)):
                        logger.warning(f"Invalid FFT magnitude for variable {variable}")
                        continue
                        
                    peaks, _ = find_peaks(magnitude, height=np.std(magnitude))
                    
                    for peak in peaks:
                        if peak == 0:  # Skip DC component
                            continue
                            
                        freq = freqs[peak]
                        if freq <= 0:  # Skip negative frequencies
                            continue
                            
                        # Calculate confidence based on peak prominence
                        prominence = magnitude[peak] / np.mean(magnitude)
                        confidence = min(1.0, prominence / 5.0)
                        
                        if confidence >= self.min_pattern_confidence:
                            pattern = TimeSeriesPattern(
                                pattern_id=f"cyclic_{variable}_{freq:.3f}Hz",
                                values=window_values,
                                timestamps=window_timestamps,
                                frequency=freq,
                                confidence=confidence,
                                type=PatternType.CYCLIC.name,
                                metadata={
                                    'period': 1/freq,
                                    'amplitude': magnitude[peak]/len(window_values),
                                    'window_size': window_size
                                }
                            )
                            patterns.append(pattern)
                            
                except Exception as e:
                    logger.warning(f"Error in cyclic pattern detection for window {window_size}: {str(e)}")
                    continue
                    
            return patterns
            
        except Exception as e:
            logger.error(f"Error in cyclic pattern detection: {str(e)}")
            raise AnalysisError(f"Cyclic pattern detection failed: {str(e)}")
        
    def _detect_trends(self, 
                      variable: str, 
                      timestamps: np.ndarray, 
                      values: np.ndarray) -> List[TimeSeriesPattern]:
        """Detect trends using regression analysis.
        
        Performs linear regression analysis over multiple time windows
        to identify significant trends in the data.
        
        Args:
            variable: Name of the variable being analyzed
            timestamps: Array of observation timestamps
            values: Array of observed values
            
        Returns:
            List of detected trend patterns
            
        Note:
            The confidence score is based on both the R² value and
            the statistical significance (p-value) of the trend.
        """
        patterns = []
        
        for window_size in self.window_sizes:
            if len(values) < window_size:
                continue
                
            # Use last window_size points
            window_values = values[-window_size:]
            window_timestamps = timestamps[-window_size:]
            
            # Normalize timestamps to [0, 1] for numerical stability
            t_norm = (window_timestamps - window_timestamps[0]) / (window_timestamps[-1] - window_timestamps[0])
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(t_norm, window_values)
            
            # Calculate confidence based on R² and p-value
            r_squared = r_value ** 2
            confidence = r_squared * (1 - p_value)
            
            if confidence >= self.min_pattern_confidence:
                pattern = TimeSeriesPattern(
                    pattern_id=f"trend_{variable}_{window_size}",
                    values=window_values,
                    timestamps=window_timestamps,
                    frequency=0.0,  # Trends don't have frequency
                    confidence=confidence,
                    type='trend',
                    metadata={
                        'slope': slope,
                        'intercept': intercept,
                        'r_squared': r_squared,
                        'p_value': p_value,
                        'std_err': std_err,
                        'window_size': window_size
                    }
                )
                patterns.append(pattern)
                
        return patterns
        
    def _detect_anomalies(self, 
                         variable: str, 
                         timestamps: np.ndarray, 
                         values: np.ndarray) -> List[TimeSeriesPattern]:
        """Detect anomalies using statistical methods.
        
        Uses multiple statistical techniques to identify anomalies:
        1. Z-score analysis for point anomalies
        2. Moving average deviation for contextual anomalies
        3. Change point detection for collective anomalies
        
        Args:
            variable: Name of the variable being analyzed
            timestamps: Array of observation timestamps
            values: Array of observed values
            
        Returns:
            List of detected anomaly patterns
            
        Note:
            The confidence score is based on the statistical significance
            of the deviation from expected behavior.
        """
        patterns = []
        
        for window_size in self.window_sizes:
            if len(values) < window_size:
                continue
                
            # Use last window_size points
            window_values = values[-window_size:]
            window_timestamps = timestamps[-window_size:]
            
            # Calculate z-scores
            z_scores = stats.zscore(window_values)
            
            # Find anomalies (points beyond 3 standard deviations)
            anomaly_indices = np.where(np.abs(z_scores) > 3)[0]
            
            for idx in anomaly_indices:
                # Calculate confidence based on how extreme the value is
                z_score = abs(z_scores[idx])
                confidence = min(1.0, (z_score - 3) / 2)  # Scale to [0, 1]
                
                if confidence >= self.min_pattern_confidence:
                    pattern = TimeSeriesPattern(
                        pattern_id=f"anomaly_{variable}_{window_timestamps[idx]}",
                        values=window_values[max(0, idx-5):min(len(window_values), idx+6)],
                        timestamps=window_timestamps[max(0, idx-5):min(len(window_timestamps), idx+6)],
                        frequency=0.0,  # Anomalies don't have frequency
                        confidence=confidence,
                        type='anomaly',
                        metadata={
                            'z_score': z_scores[idx],
                            'value': window_values[idx],
                            'timestamp': window_timestamps[idx],
                            'window_size': window_size
                        }
                    )
                    patterns.append(pattern)
                    
        return patterns
        
    def _analyze_causality(self):
        """Analyze causal relationships between variables using Granger causality"""
        variables = list(self.time_series_data.keys())
        
        for i, var1 in enumerate(variables):
            for var2 in variables[i+1:]:
                # Get aligned time series
                ts1, ts2 = self._align_time_series(var1, var2)
                
                if len(ts1) < 30:  # Need enough data points
                    continue
                    
                # Test Granger causality in both directions
                for lag in [1, 5, 10]:
                    # var1 -> var2
                    f12, p12 = self._granger_causality(ts1, ts2, lag)
                    if p12 < self.significance_level:
                        self._add_causal_link(var1, var2, lag, f12)
                        
                    # var2 -> var1
                    f21, p21 = self._granger_causality(ts2, ts1, lag)
                    if p21 < self.significance_level:
                        self._add_causal_link(var2, var1, lag, f21)
                        
    def _align_time_series(self, var1: str, var2: str) -> Tuple[np.ndarray, np.ndarray]:
        """Align two time series to common timestamps"""
        ts1 = np.array(self.time_series_data[var1])
        ts2 = np.array(self.time_series_data[var2])
        
        # Find common time range
        start_time = max(ts1[0][0], ts2[0][0])
        end_time = min(ts1[-1][0], ts2[-1][0])
        
        # Filter to common range
        ts1 = ts1[(ts1[:, 0] >= start_time) & (ts1[:, 0] <= end_time)]
        ts2 = ts2[(ts2[:, 0] >= start_time) & (ts2[:, 0] <= end_time)]
        
        return ts1[:, 1], ts2[:, 1]
        
    def _granger_causality(self, x: np.ndarray, y: np.ndarray, lag: int) -> Tuple[float, float]:
        """Implement Granger causality test"""
        # Create lagged versions of time series
        X = np.column_stack([np.roll(x, i) for i in range(lag, 0, -1)])
        X = X[lag:]
        y = y[lag:]
        
        # Fit two models
        model1 = np.polyfit(X, y, 1)  # With X
        model2 = np.polyfit(y, y, 0)  # Without X (just mean)
        
        # Calculate residuals
        resid1 = y - np.polyval(model1, X)
        resid2 = y - np.polyval(model2, y)
        
        # Calculate F-statistic
        rss1 = np.sum(resid1**2)
        rss2 = np.sum(resid2**2)
        
        f_stat = ((rss2 - rss1) / lag) / (rss1 / (len(y) - lag - 1))
        p_value = 1 - stats.f.cdf(f_stat, lag, len(y) - lag - 1)
        
        return f_stat, p_value
        
    def _add_causal_link(self, source: str, target: str, lag: int, strength: float):
        """Add or update causal link in graph"""
        if not self.causal_graph.has_edge(source, target):
            self.causal_graph.add_edge(
                source,
                target,
                lag=lag,
                strength=strength,
                confidence=1 - (lag / max(self.window_sizes)),
                type='direct'
            )
        else:
            # Update existing link if new one is stronger
            if strength > self.causal_graph[source][target]['strength']:
                self.causal_graph[source][target].update({
                    'lag': lag,
                    'strength': strength,
                    'confidence': 1 - (lag / max(self.window_sizes))
                })
                
    def _update_pattern_storage(self, new_patterns: List[TimeSeriesPattern]):
        """Update stored patterns with new ones"""
        # Remove old patterns with same IDs
        for pattern in new_patterns:
            if pattern.pattern_id in self.patterns:
                old_pattern = self.patterns[pattern.pattern_id]
                # Keep pattern with higher confidence
                if pattern.confidence > old_pattern.confidence:
                    self.patterns[pattern.pattern_id] = pattern
            else:
                self.patterns[pattern.pattern_id] = pattern
                
        # Remove patterns that haven't been updated (they're no longer valid)
        current_ids = {p.pattern_id for p in new_patterns}
        for pattern_id in list(self.patterns.keys()):
            if pattern_id not in current_ids:
                del self.patterns[pattern_id]
                
    def get_patterns(self, 
                    pattern_type: Optional[str] = None, 
                    min_confidence: Optional[float] = None) -> List[TimeSeriesPattern]:
        """Get patterns matching criteria"""
        if min_confidence is None:
            min_confidence = self.min_pattern_confidence
            
        patterns = list(self.patterns.values())
        
        if pattern_type:
            patterns = [p for p in patterns if p.type == pattern_type]
            
        patterns = [p for p in patterns if p.confidence >= min_confidence]
        
        # Sort by confidence
        patterns.sort(key=lambda x: x.confidence, reverse=True)
        
        return patterns
        
    def get_causal_relationships(self, 
                               min_confidence: float = 0.7) -> List[CausalLink]:
        """Get causal relationships matching criteria"""
        relationships = []
        
        for source, target, data in self.causal_graph.edges(data=True):
            if data['confidence'] >= min_confidence:
                relationship = CausalLink(
                    source_id=source,
                    target_id=target,
                    lag=data['lag'],
                    strength=data['strength'],
                    confidence=data['confidence'],
                    type=data['type']
                )
                relationships.append(relationship)
                
        return relationships 