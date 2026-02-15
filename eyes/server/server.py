import os
import datetime
import logging
import math  # Added for sqrt
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from flask_cors import CORS  
from flask_socketio import SocketIO
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from collections import deque, defaultdict
from user_manager import UserManager
from datetime import datetime, timezone  # Corrected import

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow cross-origin HTTP requests
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your_secret_key')
# Configure Socket.IO with improved settings
socketio = SocketIO(app, 
                  logger=True, 
                  engineio_logger=True, 
                  cors_allowed_origins="*", 
                  ping_timeout=30000,  # Increase ping timeout (in ms to match ping_interval)
                  ping_interval=15000,  # Keep ping interval for responsive connections
                  max_http_buffer_size=10e6)  # Increase buffer size for larger payloads

# Setup login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize user manager
user_manager = UserManager()

# Global state storage
device_metrics = defaultdict(dict)
device_heartbeats = {}
anomaly_log = deque(maxlen=100)
connected_devices = {}
sensor_data = {}  # For storing Jetson sensor readings
# Global state for acceleration streaks
# device_id -> {'x_streak': 0, 'y_streak': 0, 'z_streak': 0}
device_accel_streaks = defaultdict(lambda: {'x_streak': 0, 'y_streak': 0, 'z_streak': 0})

# System Configuration with defaults
system_config = {
    'thresholds': {
        'crowd': {
            'group_size': 4,  # Number of people that triggers crowd anomaly
            'duration': 5     # Seconds threshold for crowd persistence
        },
        'lab_coat': {
            'missing_duration': 3  # Seconds before triggering lab coat anomaly
        },
        'tower_light': {
            'green_duration': 30,   # Seconds of green light before anomaly
            'light_1_duration': 30, # Default duration for light 1
            'light_2_duration': 30, # Default duration for light 2
            'light_3_duration': 30, # Default duration for light 3
            'light_4_duration': 30  # Default duration for light 4
        },
        'detection': {
            'human_confidence': 0.4,
            'labcoat_confidence': 0.6,
            'light_confidence': 0.4,
            'door_confidence': 0.6
        },
        'sensors': {
            'window_size': 10,     # Number of readings to consider for moving average
            'co2': {
                'min': 400,        # Minimum acceptable CO2 level (ppm)
                'max': 1000,       # Maximum acceptable CO2 level (ppm)
                'rate_change': 50  # Maximum acceptable rate of change per reading
            },
            'temperature': {
                'min': 18,         # Minimum acceptable temperature (°C)
                'max': 25,         # Maximum acceptable temperature (°C)
                'rate_change': 2   # Maximum acceptable rate of change per reading
            },
            'humidity': {
                'min': 30,         # Minimum acceptable humidity (%)
                'max': 60,         # Maximum acceptable humidity (%)
                'rate_change': 5   # Maximum acceptable rate of change per reading
            },
            'pressure': {
                'min': 980,        # Minimum acceptable pressure (hPa)
                'max': 1020,       # Maximum acceptable pressure (hPa)
                'rate_change': 1   # Maximum acceptable rate of change per reading
            },
            'acceleration': {
                'x_threshold': 2.0,    # Max acceptable acceleration on X-axis (g)
                'y_threshold': 2.0,    # Max acceptable acceleration on Y-axis (g)
                'z_threshold': 2.0,    # Max acceptable acceleration on Z-axis (g)
                'consecutive_count_trigger': 20  # Number of consecutive readings to trigger anomaly
            }
        }
    },
    'server_internals': {  # New section for server operational parameters
        'stale_channel_check_interval_seconds': 60,  # How often to run the check
        'stale_channel_threshold_seconds': 120,      # Channel stale if no update for this duration
        'device_timeout_seconds': 15,                 # Device considered disconnected after this duration
        'device_timeout_check_interval_seconds': 3   # How often to check for device timeouts
    }
}

class User(UserMixin):
    def __init__(self, username, role):
        self.id = username
        self.role = role

@login_manager.user_loader
def load_user(username):
    role = user_manager.get_user_role(username)
    if role:
        return User(username, role)
    return None

def get_utc_timestamp():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')  # Corrected

# Helper to reset streaks for a device or a specific axis if a non-consecutive event occurs
def reset_accel_streaks_for_device(device_id, axis_to_keep_active=None):
    if axis_to_keep_active != 'x':
        device_accel_streaks[device_id]['x_streak'] = 0
    if axis_to_keep_active != 'y':
        device_accel_streaks[device_id]['y_streak'] = 0
    if axis_to_keep_active != 'z':
        device_accel_streaks[device_id]['z_streak'] = 0

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user_data = user_manager.verify_user(username, password)
        
        if user_data:
            user = User(username, user_data['role'])
            login_user(user)
            return redirect(url_for('dashboard'))
        
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def dashboard():
    return render_template('dashboard.html',
                         metrics=device_metrics,
                         heartbeats=device_heartbeats,
                         anomalies=list(anomaly_log),
                         sensor_data=sensor_data,
                         config=system_config['thresholds'])

@app.route('/update_metrics', methods=['POST'])
def update_metrics_route():
    data = request.get_json()
    if not data:
        logger.error("No JSON payload provided in /update_metrics")
        return jsonify({"status": "fail", "error": "No JSON payload provided"}), 400

    device_id = str(data.get('device_id', 'unknown'))
    channel = str(data.get('channel', 'unknown'))
    device_type = data.get('device_type', 'computer')
    
    logger.info(f"Received metrics from device {device_id}, channel {channel}: {data.get('metrics', {})}")

    metrics_payload = data.get('metrics', {})
    
    processed_metrics = {
        "people_count": metrics_payload.get('people_count', 0),
        "lab_coat_count": metrics_payload.get('lab_coat_count', 0),
        "last_updated": data.get('timestamp', get_utc_timestamp())
    }
    
    if 'green_intervals' in metrics_payload:
        processed_metrics["green_intervals"] = metrics_payload['green_intervals']
    if 'door_status' in metrics_payload:
        processed_metrics["door_status"] = metrics_payload['door_status']
    if 'tower_light' in metrics_payload:
        processed_metrics["tower_light"] = metrics_payload['tower_light']
    if 'light_1_status' in metrics_payload: # Assuming payload sends 'light_X_status'
        processed_metrics["light_1_status"] = metrics_payload['light_1_status']
    if 'light_2_status' in metrics_payload:
        processed_metrics["light_2_status"] = metrics_payload['light_2_status']
    if 'light_3_status' in metrics_payload:
        processed_metrics["light_3_status"] = metrics_payload['light_3_status']
    if 'light_4_status' in metrics_payload:
        processed_metrics["light_4_status"] = metrics_payload['light_4_status']
    if 'efficiency' in metrics_payload:
        processed_metrics["efficiency"] = metrics_payload['efficiency']
    
    for key, value in metrics_payload.items():
        if key not in processed_metrics:
            processed_metrics[key] = value

    current_sensor_data_for_device = None
    if 'sensor_data' in data:
        sensor_data[device_id] = {
            'data': data['sensor_data'],
            'timestamp': get_utc_timestamp()
        }
        current_sensor_data_for_device = sensor_data[device_id]
    else:
        sensor_categories = system_config['thresholds']['sensors']
        sensor_keys = [k for k in sensor_categories if k != 'window_size']
        sensor_metrics_from_payload = {k: metrics_payload[k] for k in sensor_keys if k in metrics_payload}
        if sensor_metrics_from_payload:
            sensor_data[device_id] = {
                'data': sensor_metrics_from_payload,
                'timestamp': get_utc_timestamp()
            }
            current_sensor_data_for_device = sensor_data[device_id]
        elif device_id in sensor_data: # Use existing if not in this payload
            current_sensor_data_for_device = sensor_data[device_id]

    if device_id not in device_metrics:
        device_metrics[device_id] = {}

    device_metrics[device_id][channel] = processed_metrics
    device_heartbeats[device_id] = get_utc_timestamp()

    # Emit Socket.IO event for the original channel
    socket_payload_original = {
        'device_id': device_id,
        'channel': channel,
        'metrics': processed_metrics
    }
    if current_sensor_data_for_device: # Include sensor data if available for this device
        socket_payload_original['sensor_data'] = current_sensor_data_for_device
    
    logger.info(f"Emitting 'update_metrics' for original channel with payload: {socket_payload_original}")
    try:
        socketio.emit('update_metrics', socket_payload_original)
        logger.info(f"Emitted 'update_metrics' for {device_id} channel {channel}")
    except Exception as e:
        logger.error(f"Error emitting 'update_metrics' via Socket.IO for original channel {channel}: {e}")

    return jsonify({"status": "success", "device_id": device_id, "channel": channel})

@app.route('/config', methods=['GET', 'POST'])
@login_required
def config():
    if request.method == 'POST':
        try:
            new_config = request.get_json()
            if not new_config:
                return jsonify({'status': 'error', 'message': 'No JSON payload provided'}), 400

            for category, settings in new_config.items():
                if category not in system_config['thresholds']:
                    continue
                for key, value in settings.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (int, float)):
                                system_config['thresholds'][category][key][sub_key] = float(sub_value)
                            elif isinstance(sub_value, str): # Allow string values for potential status or other non-numeric settings
                                system_config['thresholds'][category][key][sub_key] = sub_value
                    elif isinstance(value, (int, float)):
                        system_config['thresholds'][category][key] = float(value)
                    elif isinstance(value, str): # Allow string values for top-level settings
                         system_config['thresholds'][category][key] = value
            
            # Notify all connected clients about the configuration change
            try:
                socketio.emit('config_update', system_config['thresholds'])
            except Exception as e:
                logger.error(f"Error emitting config update: {e}")
                
            return jsonify({'status': 'success', 'config': system_config['thresholds']})
        except Exception as e:
            logger.error(f"Error updating configuration: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e)}), 400
    
    return render_template('config.html', config=system_config['thresholds'])

@app.route('/api/config')
def get_config():
    return jsonify(system_config['thresholds'])

@app.route('/api/status')
def get_status():
    """API endpoint to check server status and connected clients"""
    return jsonify({
        'status': 'ok',
        'connected_clients': len(connected_devices),
        'devices': list(device_metrics.keys()),
        'uptime': get_utc_timestamp()
    })

@socketio.on('connect')
def handle_connect():
    client_id = request.sid
    logger.info(f"Client connected: {client_id} from IP: {request.remote_addr}")
    connected_devices[client_id] = {'connected_at': get_utc_timestamp(), 'ip': request.remote_addr}
    
    # Send current configuration to newly connected clients
    try:
        socketio.emit('config_update', system_config['thresholds'], room=client_id)
    except Exception as e:
        logger.error(f"Error sending config on connect: {e}")

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    # Atomically remove the client from connected_devices and get its info
    device_info = connected_devices.pop(sid, None)

    if device_info:
        device_id = device_info.get('id') # This 'id' is now set by 'self_identify'
        
        log_message_prefix = f"Client disconnected: SID {sid}"
        if device_id:
            log_message_prefix += f", Device ID {device_id}"
        else:
            # This case means the client connected but might not have self-identified
            log_message_prefix += " (client had no associated device_id, possibly did not self-identify)"
        
        logger.info(log_message_prefix)

        if device_id: # Proceed with cleanup only if a proper device_id was associated
            # Remove from relevant data structures, checking if key existed for logging
            if device_metrics.pop(device_id, None) is not None:
                logger.info(f"Removed metrics for disconnected device {device_id} (socket disconnect)") # Clarified log
            if device_heartbeats.pop(device_id, None) is not None:
                logger.info(f"Removed heartbeat for disconnected device {device_id} (socket disconnect)") # Clarified log
            if sensor_data.pop(device_id, None) is not None:
                logger.info(f"Removed sensor data for disconnected device {device_id} (socket disconnect)") # Clarified log
            if device_accel_streaks.pop(device_id, None) is not None:
                logger.info(f"Removed acceleration streaks for disconnected device {device_id} (socket disconnect)") # Clarified log
            
            # Notify frontend to remove the device from UI
            try:
                socketio.emit('device_disconnected', {'device_id': device_id})
                logger.info(f"Emitted 'device_disconnected' for {device_id}")
            except Exception as e:
                logger.error(f"Error emitting 'device_disconnected' for {device_id}: {e}")
    else:
        # This means the sid was not in connected_devices, perhaps already handled or a spurious disconnect event
        logger.info(f"Client disconnected: SID {sid} (not found in connected_devices or already processed).")

@socketio.on('self_identify')
def handle_self_identify(data):
    sid = request.sid
    device_id = data.get('device_id')
    device_type = data.get('device_type', 'computer') # Default to 'computer' if not provided

    if not device_id:
        logger.warning(f"Received self_identify from SID {sid} without a device_id. Ignoring.")
        return

    if sid in connected_devices:
        connected_devices[sid]['id'] = device_id
        connected_devices[sid]['type'] = device_type
        # Update last seen or add other relevant info if necessary
        connected_devices[sid]['identified_at'] = get_utc_timestamp()
        logger.info(f"Device self-identified: SID {sid} as Device ID {device_id}, Type {device_type}")
        
        # Optionally, send an acknowledgement back to the client
        try:
            socketio.emit('self_identity_ack', {'device_id': device_id, 'status': 'success'}, room=sid)
        except Exception as e:
            logger.error(f"Error emitting self_identity_ack to {device_id}: {e}")
    else:
        logger.warning(f"Received self_identify from SID {sid} not found in connected_devices. This might happen if disconnect occurred rapidly after connect.")

def _actually_broadcast_anomaly(anomaly_event_data):
    """This function handles the actual broadcasting of an anomaly."""
    logger.info(f"Broadcasting official anomaly: {anomaly_event_data.get('description')} for device {anomaly_event_data.get('device_id')}")
    anomaly_log.append(anomaly_event_data)
    try:
        socketio.emit('anomaly', anomaly_event_data)
        socketio.emit('popup_notification', anomaly_event_data)
        logger.info(f"Successfully broadcasted anomaly: {anomaly_event_data.get('description')} for device {anomaly_event_data.get('device_id')}")
    except Exception as e:
        logger.error(f"Error broadcasting official anomaly via Socket.IO: {e}")

def _handle_incoming_anomaly_report(data):
    logger.info(f"Handling incoming anomaly report: {data}")
    device_id = data.get('device_id', 'unknown_device')
    original_description = data.get('description', 'No description') # Keep original description from device
    anomaly_type = data.get('type', 'general_anomaly')
    details = data.get('details', {})
    timestamp = data.get('timestamp', get_utc_timestamp())

    # Base event data, might be broadcasted if not a streakable accel anomaly or if accel logic falls through
    raw_anomaly_event_data = {
        "device_id": device_id,
        "channel": details.get('channel', data.get('channel', 'N/A')),
        "description": original_description,
        "timestamp": timestamp,
        "type": anomaly_type,
        "details": details
    }

    # Check if it's an acceleration anomaly we need to count and validate against server thresholds.
    # Expecting type='acceleration_violation', details={'axis': 'x'|'y'|'z', 'value': float, ...}
    if anomaly_type == "acceleration_violation" and \
       'axis' in details and \
       str(details.get('axis')).lower() in ['x', 'y', 'z'] and \
       'value' in details:
        
        axis = str(details['axis']).lower()
        reported_value = None
        try:
            reported_value = float(details['value'])
        except (ValueError, TypeError):
            logger.warning(f"Invalid or missing 'value' in acceleration anomaly from {device_id}. Details: {details}. "
                           f"Cannot re-validate against server thresholds. Broadcasting original anomaly and resetting device streaks.")
            _actually_broadcast_anomaly(raw_anomaly_event_data)
            reset_accel_streaks_for_device(device_id)
            return

        accel_sensor_config = system_config['thresholds'].get('sensors', {}).get('acceleration', {})
        server_axis_threshold = accel_sensor_config.get(f'{axis}_threshold')
        consecutive_trigger_count = int(accel_sensor_config.get('consecutive_count_trigger', 20))

        if server_axis_threshold is None:
            logger.error(f"Server threshold for {axis}-axis acceleration is not configured. "
                         f"Cannot process anomaly for streak. Broadcasting original and resetting device streaks.")
            _actually_broadcast_anomaly(raw_anomaly_event_data)
            reset_accel_streaks_for_device(device_id)
            return

        current_streak_key = f'{axis}_streak'

        if reported_value > server_axis_threshold:
            # This is a confirmed violation according to server's config
            device_accel_streaks[device_id][current_streak_key] += 1
            # Reset streaks for other axes on the same device, as this event is specific to 'axis'
            reset_accel_streaks_for_device(device_id, axis_to_keep_active=axis)

            streak_count = device_accel_streaks[device_id][current_streak_key]
            logger.info(f"Server-confirmed {axis}-axis acceleration violation for device {device_id} "
                        f"(Value: {reported_value}g > Server Threshold: {server_axis_threshold}g). "
                        f"Current streak: {streak_count}/{consecutive_trigger_count}.")

            if streak_count >= consecutive_trigger_count:
                logger.info(f"Consecutive anomaly threshold met for {axis}-axis acceleration on device {device_id} "
                            f"after {streak_count} readings (Server Threshold: {server_axis_threshold}g).")
                
                consecutive_anomaly_data = {
                    "device_id": device_id,
                    "channel": details.get('channel', data.get('channel', 'N/A')),
                    "description": (
                        f"Consecutive ({streak_count}) high {axis}-axis acceleration readings for device {device_id}. "
                        f"Last: {reported_value}g (Server Threshold: {server_axis_threshold}g). "
                        f"Original device alert: '{original_description}'"
                    ),
                    "timestamp": timestamp,
                    "type": "consecutive_acceleration_anomaly",
                    "details": {
                        "triggering_axis": axis,
                        "last_value_g": reported_value,
                        "server_threshold_g": server_axis_threshold,
                        "consecutive_count": streak_count,
                        "original_event_details": details # Keep original details from the device event
                    }
                }
                _actually_broadcast_anomaly(consecutive_anomaly_data)
                device_accel_streaks[device_id][current_streak_key] = 0 # Reset streak after firing
            # else: streak count not met, already logged, do nothing further for clients.
        
        else: # reported_value <= server_axis_threshold
            logger.info(f"Device-reported {axis}-axis acceleration for {device_id} (Value: {reported_value}g) "
                        f"is NOT above server threshold ({server_axis_threshold}g). Resetting {axis}-streak for this device.")
            if device_accel_streaks[device_id][current_streak_key] > 0:
                 device_accel_streaks[device_id][current_streak_key] = 0
            # This event does not contribute to a server-level anomaly, so it's not broadcasted further.
            # The original device alert is effectively suppressed by the server if it doesn't meet server criteria.

    else: # Not an acceleration_violation type we are counting/streaking, or malformed for such processing.
        if device_id in device_accel_streaks: # Only log reset if there were potential streaks
             logger.info(f"Anomaly type '{anomaly_type}' for device {device_id} is not a server-streaked acceleration type or is a different event. "
                         f"Resetting all its acceleration streaks.")
             reset_accel_streaks_for_device(device_id)
        # Broadcast the original anomaly as is.
        _actually_broadcast_anomaly(raw_anomaly_event_data)

@app.route('/anomaly', methods=['POST'])
def http_handle_anomaly():
    data = request.get_json()
    if not data:
        logger.error("No JSON payload provided in POST /anomaly")
        return jsonify({"status": "fail", "error": "No JSON payload provided"}), 400
    
    _handle_incoming_anomaly_report(data) # MODIFIED
    return jsonify({"status": "success", "message": "Anomaly report received and is being processed"})

@socketio.on('anomaly') # ADDED: Handler for anomaly events from transmitters
def handle_anomaly_event(data):
    _handle_incoming_anomaly_report(data) # MODIFIED

@socketio.on_error()
def error_handler(e):
    """Global error handler for SocketIO events"""
    logger.error(f"SocketIO error: {str(e)}")

def check_stale_channels():
    """
    Periodically checks for stale channels (channels that haven't sent updates).
    Removes stale channels and notifies clients.
    Also cleans up device_heartbeats if a device has no more active channels.
    """
    global device_metrics, device_heartbeats # Ensure we are modifying the global dicts
    
    stale_threshold_seconds = system_config.get('server_internals', {}).get('stale_channel_threshold_seconds', 120)
    now_utc = datetime.now(timezone.utc)
    channels_to_remove = []  # List to store (device_id, channel) tuples

    # Iterate over a copy of items for safe modification
    for device_id, channels_data in list(device_metrics.items()):
        for channel, metrics in list(channels_data.items()):
            last_updated_str = metrics.get('last_updated')
            if last_updated_str:
                try:
                    last_updated_dt = datetime.fromisoformat(last_updated_str.replace('Z', '+00:00'))
                    # Check if the timestamp is in the future (e.g., client clock ahead)
                    if last_updated_dt > now_utc:
                        logger.warning(f"Channel '{channel}' for device '{device_id}' has a future 'last_updated' timestamp: {last_updated_str}. Ignoring for staleness check this cycle.")
                        continue  # Skip this channel for this cycle

                    if (now_utc - last_updated_dt).total_seconds() > stale_threshold_seconds:
                        logger.info(f"Channel '{channel}' for device '{device_id}' is stale (no update for > {stale_threshold_seconds}s). Last update: {last_updated_str}")
                        channels_to_remove.append((device_id, channel))
                except ValueError as e:
                    logger.error(f"Error parsing 'last_updated' timestamp ('{last_updated_str}') for channel '{channel}', device '{device_id}': {e}. Considering it stale.")
                    channels_to_remove.append((device_id, channel))
            else:
                logger.warning(f"Channel '{channel}' for device '{device_id}' has no 'last_updated' timestamp. Considering it stale.")
                channels_to_remove.append((device_id, channel))

    for device_id, channel in channels_to_remove:
        if device_id in device_metrics and channel in device_metrics[device_id]:
            del device_metrics[device_id][channel]
            logger.info(f"Removed stale channel '{channel}' for device '{device_id}' from metrics.")
            
            try:
                socketio.emit('channel_stale', {'device_id': device_id, 'channel': channel})
                logger.info(f"Emitted 'channel_stale' for device {device_id}, channel {channel}")
            except Exception as e:
                logger.error(f"Error emitting 'channel_stale' for {device_id}, channel {channel}: {e}")

            # If the device has no more channels after this removal, remove the device entry itself from device_metrics
            if not device_metrics[device_id]:                
                del device_metrics[device_id]
                logger.info(f"Device '{device_id}' has no more active channels and was removed from device_metrics.")
                # Also remove from device_heartbeats if it has no more channels
                if device_heartbeats.pop(device_id, None):
                    logger.info(f"Also removed device '{device_id}' from device_heartbeats as all its channels became stale.")

def check_device_timeouts():
    """
    Checks for devices that haven't sent a heartbeat in a while and removes them.
    """
    now = datetime.now(timezone.utc)
    timeout_seconds = system_config['server_internals']['device_timeout_seconds']
    
    # Iterate over a copy of device_ids since we might modify the dict
    timed_out_devices = []
    for device_id, last_heartbeat_str in list(device_heartbeats.items()):
        try:
            last_heartbeat_dt = datetime.fromisoformat(last_heartbeat_str.replace('Z', '+00:00'))
            if (now - last_heartbeat_dt).total_seconds() > timeout_seconds:
                timed_out_devices.append(device_id)
        except ValueError:
            logger.error(f"Could not parse heartbeat timestamp '{last_heartbeat_str}' for device {device_id}. Skipping timeout check for this device.")
            continue # Skip this device if timestamp is invalid

    for device_id in timed_out_devices:
        logger.info(f"Device {device_id} timed out (no heartbeat for > {timeout_seconds}s). Removing from system.")
        
        # Remove device data
        if device_metrics.pop(device_id, None) is not None:
            logger.info(f"Removed metrics for timed-out device {device_id}")
        if device_heartbeats.pop(device_id, None) is not None: # This should always be true as we iterated it
            logger.info(f"Removed heartbeat for timed-out device {device_id}")
        if sensor_data.pop(device_id, None) is not None:
            logger.info(f"Removed sensor data for timed-out device {device_id}")
        if device_accel_streaks.pop(device_id, None) is not None:
            logger.info(f"Removed acceleration streaks for timed-out device {device_id}")

        # Notify UI
        try:
            socketio.emit('device_disconnected', {'device_id': device_id})
            logger.info(f"Emitted 'device_disconnected' for timed-out device {device_id}")
        except Exception as e:
            logger.error(f"Error emitting 'device_disconnected' for timed-out device {device_id}: {e}")

def periodic_staleness_check_task():
    """
    Background task wrapper for check_stale_channels.
    """
    check_interval = system_config.get('server_internals', {}).get('stale_channel_check_interval_seconds', 60)
    stale_threshold = system_config.get('server_internals', {}).get('stale_channel_threshold_seconds', 120)
    logger.info(f"Starting periodic staleness check. Interval: {check_interval}s, Threshold: {stale_threshold}s.")
    while True:
        try:
            socketio.sleep(check_interval) # Use socketio.sleep for background tasks
            with app.app_context():      # Ensure app context for operations
                check_stale_channels()
        except Exception as e:
            logger.error(f"Error in periodic_staleness_check_task: {e}", exc_info=True)
            # Sleep even if an error occurs to prevent rapid looping and log spam
            socketio.sleep(check_interval) 

def periodic_device_timeout_check_task():
    """
    Background task to periodically check for device timeouts.
    """
    check_interval = system_config['server_internals']['device_timeout_check_interval_seconds']
    logger.info(f"Starting periodic device timeout check task. Interval: {check_interval}s")
    while True:
        try:
            check_device_timeouts()
        except Exception as e:
            logger.error(f"Error in periodic_device_timeout_check_task: {e}", exc_info=True)
        socketio.sleep(check_interval)

if __name__ == '__main__':
    # Create default admin user if none exists
    if not user_manager.get_user_role('admin'):
        user_manager.add_user('admin', 'admin123', 'admin')
    
    # Start the background task for checking stale channels
    socketio.start_background_task(target=periodic_staleness_check_task)
    # Start the periodic device timeout check
    socketio.start_background_task(target=periodic_device_timeout_check_task)
    
    logger.info("Starting Flask-SocketIO server...")
    # Use 0.0.0.0 to make the server accessible externally, not just localhost
    # Ensure debug=False in a production-like environment or when not actively debugging Flask itself
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)

