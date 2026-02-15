# main.py
import os
import sys
from pathlib import Path
import serial  # New import for sensor communication
import threading  # New import for running sensor logic in background
from collections import defaultdict, deque  # Added for anomaly tracking
import math  # Added for sqrt in acceleration
from datetime import datetime, timezone  # Added for UTC timestamps
import logging  # Added for logging sensor readings
import time  # Added for periodic sending loop
import configparser  # Added import for reading config.ini

# Setup logging before starting sensor thread
SCRIPT_DIR = Path(__file__).resolve().parent
LOG_FILE = SCRIPT_DIR / 'eyes' / 'logs' / 'transmitter.log'
# Create logs directory if it doesn't exist
os.makedirs(LOG_FILE.parent, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logging.info("Transmitter logging initialized, writing to %s", LOG_FILE)

# --- Setup package imports ---
EYES_DIR = SCRIPT_DIR / 'eyes'

# Add parent directory to Python path so we can import the eyes package
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO
from eyes.gui.main_window import start_gui
from eyes import config
from eyes.config.device_config import DeviceConfig

# --- Sensor Reading Function ---

# Constants for sensor anomaly detection
MAX_ANOMALY_READINGS = 20
NORMAL_STREAK_RESET_COUNT = 10

# Configuration
CONFIG_FILE = 'config.ini'
DEFAULT_CONFIG = {
    'JetsonDefaults': {
        'uart_port': '/dev/ttyTHS1',
        'baud_rate': 115200,
        'uart_timeout': 2,
        'log_level': 'INFO',
        'max_buffer_size': '5',
        'sensor_data_send_interval': '2',  # Reduced from 5 to 2 seconds
        'device_id_file': 'device_id.json',
        'server_host': '192.168.3.189',
        'server_port': '5000'
    }
}

# Global variable for sensor data send interval, to be loaded from config
SENSOR_DATA_SEND_INTERVAL = 2  # Default, will be updated from config

# Utility for UTC timestamps
def get_utc_timestamp_transmitter_side():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')

# Global state for sensor anomaly counts, this is a simplified approach.
sensor_anomaly_event_counts = defaultdict(int)

def manage_sensor_input(helpers_dict, lock):
    logging.info("manage_sensor_input thread started")
    global sensor_anomaly_event_counts  # Allow modification of the global counter

    # Safely access sensor configuration
    device_config_obj = helpers_dict.get('device_config')
    sensors_enabled = False  # Default to False
    uart_port = '/dev/ttyTHS1'  # Default UART port
    baud_rate = 115200      # Default baud rate

    if device_config_obj and hasattr(device_config_obj, 'sensor_config') and isinstance(device_config_obj.sensor_config, dict):
        sensor_cfg = device_config_obj.sensor_config
        uart_port = sensor_cfg.get('uart_port', uart_port)
        baud_rate = sensor_cfg.get('baud_rate', baud_rate)
        sensors_enabled = sensor_cfg.get('enabled', False)
        logging.info(f"Sensor Manager: Found device_config. Sensors enabled: {sensors_enabled}")
    else:
        logging.warning("Sensor Manager: Device configuration or sensor_config not found/invalid in helpers. Assuming sensors disabled.")
        # sensors_enabled remains False, uart_port and baud_rate use defaults which won't be used.

    if not sensors_enabled:
        logging.info("Sensor Manager: Sensors are disabled (either not on Jetson, explicitly disabled, or config missing). Sensor data collection will not be attempted.")
        with lock:
            # Ensure sensor_readings is an empty dict as expected
            if 'sensor_readings' not in helpers_dict or not isinstance(helpers_dict['sensor_readings'], dict):
                helpers_dict['sensor_readings'] = {}
            helpers_dict['sensor_readings_error'] = "Sensors disabled (config missing/invalid or not on Jetson)."
        return  # Exit the thread function if sensors are not enabled

    sensor_values_accumulator = []
    current_sensor_state = None  # 0: CO2, 1: Humidity, 2: TMP Temp, 3: Accelerometer, 4: Pressure

    # Default/fallback thresholds, updated for new acceleration structure
    default_sensor_thresholds = {
        'co2': {'min': 400, 'max': 1000, 'rate_change': 50},
        'humidity': {'min': 30, 'max': 60, 'rate_change': 5},
        'temperature': {'min': 18, 'max': 25, 'rate_change': 2},
        'pressure': {'min': 980, 'max': 1020, 'rate_change': 1},
        'acceleration': {  # Updated structure
            'x_threshold': 2.0,
            'y_threshold': 2.0,
            'z_threshold': 2.0
        }
    }
    device_id = helpers_dict.get('device_config').device_id if hasattr(helpers_dict.get('device_config'), 'device_id') else "unknown_transmitter"

    try:
        ser = serial.Serial(uart_port, baud_rate, timeout=1)
        logging.info(f"Sensor Manager: Listening on {uart_port} at {baud_rate} baud...")
        with lock:
            helpers_dict['sensor_readings_error'] = None  # Clear any previous error

        while True:  # This loop runs indefinitely in its thread
            if ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                logging.debug(f"RAW UART line: '{line}'")

                # Determine initial state if unknown
                if current_sensor_state is None:
                    if "PPM" in line:
                        current_sensor_state = 0
                        logging.debug("Sensor Manager: State changed to 0 (CO2)")
                    elif "H" in line and "mg" not in line:
                        current_sensor_state = 1
                        logging.debug("Sensor Manager: State changed to 1 (Humidity)")
                    elif line.endswith("C") or "TMP101" in line:
                        current_sensor_state = 2
                        logging.debug("Sensor Manager: State changed to 2 (Temperature)")
                    elif "mg" in line:  # Should be specific enough for accelerometer
                        current_sensor_state = 3
                        logging.debug("Sensor Manager: State changed to 3 (Accelerometer)")
                    elif "hPa" in line:  # If pressure comes first
                        current_sensor_state = 4
                        logging.debug("Sensor Manager: State changed to 4 (Pressure)")
                    else:
                        logging.debug(f"Sensor Manager: Unknown line for initial state: '{line}'")
                        continue

                try:
                    if current_sensor_state == 0 and "PPM" in line:
                        value = float(line.split()[0])
                        sensor_values_accumulator.append(value)
                        logging.debug(f"Sensor Manager: Parsed CO2. Accumulator: {sensor_values_accumulator}")
                        current_sensor_state = 1
                        logging.debug("Sensor Manager: State changed to 1 (Humidity)")

                    elif current_sensor_state == 1 and "H" in line and "mg" not in line:
                        value = float(line.split()[0])
                        sensor_values_accumulator.append(value)
                        logging.debug(f"Sensor Manager: Parsed Humidity. Accumulator: {sensor_values_accumulator}")
                        current_sensor_state = 2
                        logging.debug("Sensor Manager: State changed to 2 (Temperature)")

                    elif current_sensor_state == 2 and ("C" in line or "TMP101" in line):
                        tokens = line.split()
                        value = float(tokens[-2]) if "TMP101" in line else float(tokens[0])
                        sensor_values_accumulator.append(value)
                        logging.debug(f"Sensor Manager: Parsed Temperature. Accumulator: {sensor_values_accumulator}")
                        current_sensor_state = 3
                        logging.debug("Sensor Manager: State changed to 3 (Accelerometer)")

                    elif current_sensor_state == 3 and "mg" in line:
                        acc_values = []
                        parts = line.split(',')
                        for part in parts:
                            try:
                                acc_values.append(float(part.split(':')[1].strip().split()[0]))
                            except Exception as parse_err:
                                logging.warning(f"Sensor Manager: Could not parse accelerometer part: '{part}' due to {parse_err}. Defaulting to 0.")
                                acc_values.append(0)  # Default to 0 if parsing fails
                        while len(acc_values) < 3:
                            acc_values.append(0)  # Ensure 3 accelerometer values
                        sensor_values_accumulator.extend(acc_values[:3])
                        logging.debug(f"Sensor Manager: Parsed Accelerometer. Accumulator: {sensor_values_accumulator}")
                        current_sensor_state = 4
                        logging.debug("Sensor Manager: State changed to 4 (Pressure)")

                    elif current_sensor_state == 4 and "hPa" in line:  # Modified condition for pressure
                        try:
                            value = float(line.split()[0])  # Assumes format "VALUE hPa"
                        except (ValueError, IndexError) as parse_err:
                            logging.warning(f"Sensor Manager: Could not parse pressure from line: '{line}' due to {parse_err}. Defaulting to 0.")
                            value = 0  # Default to 0 if parsing fails
                        sensor_values_accumulator.append(value)
                        logging.debug(f"Sensor Manager: Parsed Pressure. Accumulator: {sensor_values_accumulator}")

                        # A complete cycle of sensor data is received
                        with lock:
                            logging.info(f"Sensor Manager: Attempting to finalize data. Accumulator length: {len(sensor_values_accumulator)}. Data: {sensor_values_accumulator}")
                            parsed_sensor_data = {}
                            if len(sensor_values_accumulator) == 7:  # Expected: co2, hum, temp, ax, ay, az, pres
                                parsed_sensor_data = {
                                    "co2": sensor_values_accumulator[0],
                                    "humidity": sensor_values_accumulator[1],
                                    "temperature": sensor_values_accumulator[2],
                                    "acceleration": [sensor_values_accumulator[3], sensor_values_accumulator[4], sensor_values_accumulator[5]],
                                    "pressure": sensor_values_accumulator[6]
                                }
                                helpers_dict['sensor_readings'] = parsed_sensor_data.copy()
                                logging.info(f"Sensor readings updated (7 values): {parsed_sensor_data}")
                                helpers_dict['sensor_readings_error'] = None
                            elif len(sensor_values_accumulator) == 5:  # Expected: temp, ax, ay, az, pres
                                logging.info("Sensor Manager: Detected 5 values. Assuming Temp, Accel, Pressure cycle.")
                                parsed_sensor_data = {
                                    "co2": None,
                                    "humidity": None,
                                    "temperature": sensor_values_accumulator[0],
                                    "acceleration": [sensor_values_accumulator[1], sensor_values_accumulator[2], sensor_values_accumulator[3]],
                                    "pressure": sensor_values_accumulator[4]
                                }
                                helpers_dict['sensor_readings'] = parsed_sensor_data.copy()
                                logging.info(f"Sensor readings updated (5 values): {parsed_sensor_data}")
                                helpers_dict['sensor_readings_error'] = None
                            else:
                                logging.warning(f"Sensor Manager: Unexpected number of values ({len(sensor_values_accumulator)}): {sensor_values_accumulator}. Clearing buffer.")
                                helpers_dict['sensor_readings_error'] = f"Unexpected number of values: {len(sensor_values_accumulator)}"
                            sensor_values_accumulator.clear()
                            current_sensor_state = None  # Reset state after processing a full cycle or on error

                        # --- ANOMALY DETECTION LOGIC FOR SENSORS ---
                        accel_data = parsed_sensor_data.get("acceleration")
                        if accel_data and isinstance(accel_data, list) and len(accel_data) == 3:
                            ax, ay, az = accel_data

                            current_config = helpers_dict.get('device_config')
                            sensor_config_from_server = {}
                            if current_config and hasattr(current_config, 'sensor_config') and isinstance(current_config.sensor_config, dict):
                                sensor_config_from_server = current_config.sensor_config
                            logging.debug(f"Sensor Manager: Anomaly detection using sensor_config from device_config: {sensor_config_from_server}")

                            accel_thresholds_specific = sensor_config_from_server.get('acceleration', {})
                            logging.debug(f"Sensor Manager: Specific accel thresholds from server config: {accel_thresholds_specific}")
                            default_accel_conf = default_sensor_thresholds.get('acceleration', {})

                            x_thresh = accel_thresholds_specific.get('x_threshold', default_accel_conf.get('x_threshold', 2.0))
                            y_thresh = accel_thresholds_specific.get('y_threshold', default_accel_conf.get('y_threshold', 2.0))
                            z_thresh = accel_thresholds_specific.get('z_threshold', default_accel_conf.get('z_threshold', 2.0))
                            logging.info(f"Sensor Manager: Using accel thresholds for anomaly detection - X: {x_thresh}g, Y: {y_thresh}g, Z: {z_thresh}g")

                            anomalous_axes_details_list = []
                            is_accel_anomaly = False

                            if abs(ax) > x_thresh:
                                anomalous_axes_details_list.append(f"X-axis: {ax:.2f}g (Threshold: {x_thresh}g)")
                                is_accel_anomaly = True
                            if abs(ay) > y_thresh:
                                anomalous_axes_details_list.append(f"Y-axis: {ay:.2f}g (Threshold: {y_thresh}g)")
                                is_accel_anomaly = True
                            if abs(az) > z_thresh:
                                anomalous_axes_details_list.append(f"Z-axis: {az:.2f}g (Threshold: {z_thresh}g)")
                                is_accel_anomaly = True

                            if is_accel_anomaly:
                                helpers_dict['sensor_anomaly_data']['streaks']['acceleration'] += 1
                                current_streak = helpers_dict['sensor_anomaly_data']['streaks']['acceleration']
                                logging.info(f"Sensor Manager: Accelerometer anomaly detected. Current streak: {current_streak}/{MAX_ANOMALY_READINGS}")

                                if current_streak == MAX_ANOMALY_READINGS:
                                    anomaly_description = (f"Sensor acceleration on device {device_id} has recorded {MAX_ANOMALY_READINGS} consecutive anomalous readings. "
                                                           f"Details: {'; '.join(anomalous_axes_details_list)}")

                                    anomaly_payload = {
                                        "ax": ax, "ay": ay, "az": az,
                                        "x_threshold_used": x_thresh,
                                        "y_threshold_used": y_thresh,
                                        "z_threshold_used": z_thresh,
                                        "anomalous_axes_report": anomalous_axes_details_list,
                                        "sensor_type": "acceleration",
                                        "num_consecutive_anomalous_readings": MAX_ANOMALY_READINGS
                                    }

                                    if 'vthread' in helpers_dict and hasattr(helpers_dict['vthread'], 'send_anomaly_to_server'):
                                        logging.info(f"Sensor Manager: Sending {MAX_ANOMALY_READINGS} consecutive acceleration anomaly: {anomaly_description}")
                                        helpers_dict['vthread'].send_anomaly_to_server(
                                            description=anomaly_description,
                                            anomaly_type="sensor_acceleration_consecutive",
                                            details=anomaly_payload
                                        )
                                    else:
                                        logging.warning("Sensor Manager: 'vthread' (Display_Manager) not found or send_anomaly_to_server missing. Cannot send acceleration anomaly.")
                                    # Reset streak after reporting
                                    helpers_dict['sensor_anomaly_data']['streaks']['acceleration'] = 0
                                    logging.info(f"Sensor Manager: Acceleration anomaly streak reset after reporting.")
                            else: # Not an anomaly
                                if helpers_dict['sensor_anomaly_data']['streaks']['acceleration'] > 0:
                                    logging.info(f"Sensor Manager: Acceleration readings normal. Resetting streak from {helpers_dict['sensor_anomaly_data']['streaks']['acceleration']}.")
                                    helpers_dict['sensor_anomaly_data']['streaks']['acceleration'] = 0

                        # --- ANOMALY DETECTION FOR OTHER SENSORS (CO2, Humidity, Temperature, Pressure) ---
                        sensor_types_to_check = {
                            "co2": {"unit": "ppm", "fields": ["min", "max"]},
                            "humidity": {"unit": "%", "fields": ["min", "max"]},
                            "temperature": {"unit": "°C", "fields": ["min", "max"]},
                            "pressure": {"unit": "hPa", "fields": ["min", "max"]}
                        }

                        current_config = helpers_dict.get('device_config')
                        sensor_config_from_server = {}
                        if current_config and hasattr(current_config, 'sensor_config') and isinstance(current_config.sensor_config, dict):
                            sensor_config_from_server = current_config.sensor_config
                        
                        logging.debug(f"Sensor Manager: Anomaly detection for other sensors using sensor_config from device_config: {sensor_config_from_server}")

                        for sensor_name, sensor_details in sensor_types_to_check.items():
                            sensor_value = parsed_sensor_data.get(sensor_name)
                            if sensor_value is None:
                                continue

                            # Get thresholds from server config or fall back to defaults
                            specific_thresholds = sensor_config_from_server.get(sensor_name, {})
                            default_thresholds = default_sensor_thresholds.get(sensor_name, {})

                            min_thresh = specific_thresholds.get('min', default_thresholds.get('min'))
                            max_thresh = specific_thresholds.get('max', default_thresholds.get('max'))
                            
                            logging.debug(f"Sensor Manager: Thresholds for {sensor_name} - Min: {min_thresh}, Max: {max_thresh} (using server config: {sensor_name in sensor_config_from_server})")

                            is_anomaly = False
                            anomaly_condition_desc = ""

                            if min_thresh is not None and sensor_value < min_thresh:
                                is_anomaly = True
                                anomaly_condition_desc = f"Value {sensor_value}{sensor_details['unit']} < Min Threshold {min_thresh}{sensor_details['unit']}"
                            elif max_thresh is not None and sensor_value > max_thresh:
                                is_anomaly = True
                                anomaly_condition_desc = f"Value {sensor_value}{sensor_details['unit']} > Max Threshold {max_thresh}{sensor_details['unit']}"

                            if is_anomaly:
                                helpers_dict['sensor_anomaly_data']['streaks'][sensor_name] += 1
                                current_streak = helpers_dict['sensor_anomaly_data']['streaks'][sensor_name]
                                logging.info(f"Sensor Manager: {sensor_name.capitalize()} anomaly detected. Streak: {current_streak}/{MAX_ANOMALY_READINGS}. {anomaly_condition_desc}")

                                if current_streak >= MAX_ANOMALY_READINGS: # Use >= for safety
                                    anomaly_description = (
                                        f"Sensor {sensor_name} on device {device_id} has recorded {MAX_ANOMALY_READINGS} "
                                        f"consecutive anomalous readings. Last reading: {anomaly_condition_desc}"
                                    )
                                    anomaly_payload = {
                                        "sensor_type": sensor_name,
                                        "value": sensor_value,
                                        "min_threshold_used": min_thresh,
                                        "max_threshold_used": max_thresh,
                                        "condition": anomaly_condition_desc,
                                        "num_consecutive_anomalous_readings": MAX_ANOMALY_READINGS
                                    }

                                    if 'vthread' in helpers_dict and hasattr(helpers_dict['vthread'], 'send_anomaly_to_server'):
                                        logging.info(f"Sensor Manager: Sending {MAX_ANOMALY_READINGS} consecutive {sensor_name} anomaly: {anomaly_description}")
                                        helpers_dict['vthread'].send_anomaly_to_server(
                                            description=anomaly_description,
                                            anomaly_type=f"sensor_{sensor_name}_consecutive",
                                            details=anomaly_payload
                                        )
                                    else:
                                        logging.warning(f"Sensor Manager: 'vthread' (Display_Manager) not found or send_anomaly_to_server missing. Cannot send {sensor_name} anomaly.")
                                    
                                    helpers_dict['sensor_anomaly_data']['streaks'][sensor_name] = 0 # Reset streak
                                    logging.info(f"Sensor Manager: {sensor_name.capitalize()} anomaly streak reset after reporting.")
                            else: # Not an anomaly for this sensor
                                if helpers_dict['sensor_anomaly_data']['streaks'][sensor_name] > 0:
                                    logging.info(f"Sensor Manager: {sensor_name.capitalize()} readings normal. Resetting streak from {helpers_dict['sensor_anomaly_data']['streaks'][sensor_name]}.")
                                    helpers_dict['sensor_anomaly_data']['streaks'][sensor_name] = 0
                        # --- END OF ANOMALY DETECTION FOR OTHER SENSORS ---

                    elif line:
                        logging.debug(f"Sensor Manager: Incomplete line or waiting for more data: '{line}'")

                except Exception as e_parse:  # ADDED/RESTORED inner try-except block
                    logging.error(f"Sensor Manager: Error parsing sensor data: {e_parse} on line: '{line}'. Clearing buffer.")
                    sensor_values_accumulator.clear()
                    current_sensor_state = None
                    with lock:
                        helpers_dict['sensor_readings_error'] = f"Parsing error: {e_parse}"
            else:  # No data in serial buffer
                time.sleep(0.1)

    except serial.SerialException as se:
        error_msg = f"Sensor Manager: Serial port error on {uart_port}: {se}. Sensor data will not be available."
        logging.error(error_msg)
        with lock:
            helpers_dict['sensor_readings_error'] = error_msg
            helpers_dict['sensor_readings'] = []  # Clear readings on serial error
    except Exception as e:
        error_msg = f"Sensor Manager: Unexpected error: {e}"
        logging.error(error_msg)
        with lock:
            helpers_dict['sensor_readings_error'] = error_msg
            helpers_dict['sensor_readings'] = []  # Clear readings on unexpected error
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            logging.info(f"Sensor Manager: Closed serial port {uart_port}.")

# --- Initialize device configuration ---
device_config = DeviceConfig()

# --- Device Configuration ---
# Correctly determine torch device based on availability
device_torch = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device_config.device_type} (using torch device: {device_torch})")

# Correctly update the SENSOR_DATA_SEND_INTERVAL from loaded config
# The variable SENSOR_DATA_SEND_INTERVAL is already global as it was defined at the module level earlier.

# Create a ConfigParser instance to read the config file
cfg_parser = configparser.ConfigParser()
# CONFIG_FILE is defined globally in this script as 'config.ini'
cfg_parser.read(CONFIG_FILE)

# Use the parser to get the value
loaded_interval = cfg_parser.get('JetsonDefaults', 'sensor_data_send_interval',
                              fallback=DEFAULT_CONFIG['JetsonDefaults']['sensor_data_send_interval'])
try:
    SENSOR_DATA_SEND_INTERVAL = int(loaded_interval)
    logging.info(f"Sensor data send interval set to {SENSOR_DATA_SEND_INTERVAL} seconds from config.")
except ValueError:
    SENSOR_DATA_SEND_INTERVAL = int(DEFAULT_CONFIG['JetsonDefaults']['sensor_data_send_interval'])
    logging.error(f"Invalid sensor_data_send_interval '{loaded_interval}' in config. Using default {SENSOR_DATA_SEND_INTERVAL} seconds.")

# **Load Trained Door Model**
door_model_path = device_config.models_dir / device_config.door_model_filename
door_model = models.mobilenet_v2(pretrained=False)
door_model.classifier[1] = nn.Linear(door_model.last_channel, 2)
door_model.load_state_dict(torch.load(str(door_model_path), map_location=device_torch))
door_model.to(device_torch)
door_model.eval()

# **Load YOLO Models**
human_model_path = device_config.models_dir / device_config.yolo_human_model_filename
human_model = YOLO(str(human_model_path)).to(device_torch)

light_model_path = device_config.models_dir / device_config.light_model_filename
light_model = YOLO(str(light_model_path)).to(device_torch)

labcoat_model_path = device_config.models_dir / device_config.labcoat_model_filename
labcoat_model = YOLO(str(labcoat_model_path)).to(device_torch)

object_model_path = device_config.models_dir / device_config.object_model_filename
object_model = YOLO(str(object_model_path)).to(device_torch)

# --- Initialize Helpers Dictionary from Config ---
helpers = {**config.HELPERS_STATIC_CONFIG}
helpers['sensor_readings'] = {}  # Initialize as dict for structured sensor data
helpers['sensor_readings_error'] = None  # Initialize error state for sensor
helpers['sensor_anomaly_data'] = {
    'lists': defaultdict(lambda: deque(maxlen=MAX_ANOMALY_READINGS)),
    'streaks': defaultdict(int),
    'anomaly_event_queue': deque()
}
sensor_data_lock = threading.Lock()  # Lock for safe updates to helpers dict

# Add loaded models and device configuration to helpers
helpers.update({
    "human_model": human_model,
    "door_model": door_model,
    "light_model": light_model,
    "labcoat_model": labcoat_model,
    "object_model": object_model,  # Added for the new object exit model
    "device_config": device_config
})

# Adjust ROIs based on STREAM mode from config
if config.STREAM == 1: # Stream mode
    helpers["turnstile_rois"] = helpers.pop("turnstile_rois_stream_mode", [])
    helpers["light_rois"] = helpers.pop("light_rois_stream_mode", [])
else: # File mode
    helpers["turnstile_rois"] = helpers.pop("turnstile_rois_files_mode", [])
    helpers["light_rois"] = helpers.pop("light_rois_files_mode", [])

# door_roi and expanded_roi are not mode-dependent in the provided config, load them directly.
helpers["door_roi"] = helpers.pop("door_roi", [0,0,1,1]) # Default if not found
helpers["expanded_roi"] = helpers.pop("expanded_roi", [0,0,1,1]) # Default if not found

# Ensure ROIs are always present, even if empty or defaulted, to prevent key errors

# --- Determine Video Sources based on config ---
if config.STREAM == 0:
    # For file paths, join with EYES_DIR since paths in config are relative to package root
    STANDARD_VIDEO_SOURCES = [str(EYES_DIR / p) for p in config.STANDARD_VIDEO_FILES]
    TOWER_VIDEO_SOURCES = [str(EYES_DIR / p) for p in config.TOWER_VIDEO_FILES]
    ENTRANCE_VIDEO_SOURCES = [str(EYES_DIR / p) for p in config.ENTRANCE_VIDEO_FILES]
    EXIT_VIDEO_SOURCES = [str(EYES_DIR / p) for p in config.EXIT_VIDEO_FILES]
else:  # STREAM == 1
    # RTSP sources are absolute URLs, no change needed
    STANDARD_VIDEO_SOURCES = config.STANDARD_RTSP_SOURCES
    TOWER_VIDEO_SOURCES = config.TOWER_RTSP_SOURCES
    ENTRANCE_VIDEO_SOURCES = config.ENTRANCE_RTSP_SOURCES
    EXIT_VIDEO_SOURCES = config.EXIT_RTSP_SOURCES

VIDEO_SOURCES = (
    [{"type": "standard", "path": src} for src in STANDARD_VIDEO_SOURCES] +
    [{"type": "entrance", "path": src} for src in ENTRANCE_VIDEO_SOURCES] +
    [{"type": "tower", "path": src} for src in TOWER_VIDEO_SOURCES] +
    [{"type": "exit", "path": src} for src in EXIT_VIDEO_SOURCES]
)

# --- Start Sensor Management Thread ---
# This thread will run in the background, updating helpers['sensor_readings']
sensor_manager_thread = threading.Thread(
    target=manage_sensor_input,
    args=(helpers, sensor_data_lock),  # Pass helpers dict and lock to the thread
    daemon=True  # Ensures thread exits when main program exits
)
sensor_manager_thread.start()
logging.info("Sensor management thread initiated.")

# Start the GUI with device-specific configuration
start_gui(VIDEO_SOURCES, helpers, config.STREAM)

# Start periodic sensor data sending if in headless mode and sensor manager is active
if config.HEADLESS_MODE and sensor_manager_thread.is_alive():
    logging.info(f"Headless mode: Starting periodic sensor data sending every {SENSOR_DATA_SEND_INTERVAL} seconds.")
    def periodic_sender_thread_func(event):
        next_send_time = time.time()
        while not event.is_set():
            current_time = time.time()
            if current_time >= next_send_time:
                # Replace with actual sensor data sending logic
                logging.info("Sending sensor data...")
                next_send_time = current_time + SENSOR_DATA_SEND_INTERVAL
            sleep_duration = max(0, next_send_time - time.time())
            time.sleep(min(sleep_duration, 1))  # Sleep at most 1s, or less if next send is sooner

    stop_event = threading.Event()
    sender_thread = threading.Thread(target=periodic_sender_thread_func, args=(stop_event,))
    sender_thread.daemon = True  # Ensure thread exits when main program exits
    sender_thread.start()
elif config.HEADLESS_MODE and not sensor_manager_thread.is_alive():
    logging.warning("Headless mode: Sensor manager not active, cannot send sensor data.")

