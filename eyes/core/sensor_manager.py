"""Sensor manager for Jetson Nano Orin."""
import threading
import time
import logging
from collections import defaultdict, deque

class SensorManager:
    def __init__(self, config):
        self._config = config
        self._running = False
        self._readings = {}
        self._history = defaultdict(lambda: deque(maxlen=30))  # 30 seconds of history
        self._lock = threading.Lock()
        self._last_error_time = defaultdict(float)
        self._error_cooldown = 60  # 60 seconds between repeated error reports
        self._recovery_attempts = defaultdict(int)
        self._max_recovery_attempts = 3

    def start(self):
        self._running = True
        self._sensor_thread = threading.Thread(target=self._sensor_loop)
        self._sensor_thread.daemon = True
        self._sensor_thread.start()

    def stop(self):
        self._running = False
        if hasattr(self, '_sensor_thread'):
            self._sensor_thread.join(timeout=1.0)

    def _sensor_loop(self):
        while self._running:
            try:
                for sensor_id, sensor_config in self._config.items():
                    try:
                        reading = self._read_sensor(sensor_id, sensor_config)
                        with self._lock:
                            self._readings[sensor_id] = reading
                            self._history[sensor_id].append(reading)
                            self._recovery_attempts[sensor_id] = 0  # Reset on success
                    except Exception as e:
                        self._handle_sensor_error(sensor_id, e)
                
                time.sleep(1.0)  # Sample rate
            except Exception as e:
                logging.error(f"Error in sensor loop: {e}")
                time.sleep(5.0)  # Back off on error

    def _handle_sensor_error(self, sensor_id, error):
        current_time = time.time()
        last_error = self._last_error_time[sensor_id]
        
        # Only log errors if we haven't seen one recently
        if current_time - last_error > self._error_cooldown:
            logging.error(f"Sensor {sensor_id} error: {error}")
            self._last_error_time[sensor_id] = current_time
        
        # Increment recovery attempts
        self._recovery_attempts[sensor_id] += 1
        
        # If we've tried too many times, mark sensor as failed
        if self._recovery_attempts[sensor_id] >= self._max_recovery_attempts:
            with self._lock:
                self._readings[sensor_id] = {"status": "failed", "error": str(error)}
        
        time.sleep(2.0)  # Back off on error

    def get_latest_readings(self):
        with self._lock:
            readings = self._readings.copy()
            anomalies = self._check_anomalies()
            if anomalies:
                readings['anomalies'] = anomalies
            return readings

    def _check_anomalies(self):
        anomalies = {}
        for sensor_id, history in self._history.items():
            if not history:
                continue
                
            config = self._config.get(sensor_id, {})
            thresholds = config.get('thresholds', {})
            
            # Get current and historical values
            current = history[-1]
            if len(history) >= 2:
                rate_change = (current - history[-2]) / 1.0  # per second
            else:
                rate_change = 0
                
            # Check thresholds
            if 'min' in thresholds and current < thresholds['min']:
                anomalies[sensor_id] = {
                    'threshold_violation': 'min',
                    'value': current,
                    'moving_average': sum(history)/len(history)
                }
            elif 'max' in thresholds and current > thresholds['max']:
                anomalies[sensor_id] = {
                    'threshold_violation': 'max',
                    'value': current,
                    'moving_average': sum(history)/len(history)
                }
            elif 'rate' in thresholds and abs(rate_change) > thresholds['rate']:
                anomalies[sensor_id] = {
                    'threshold_violation': 'rate',
                    'value': current,
                    'rate_change': rate_change,
                    'moving_average': sum(history)/len(history)
                }
                
        return anomalies

    def update_config(self, new_config):
        """Update sensor configuration"""
        with self._lock:
            self._config.update(new_config)