# display_manager.py
from PyQt5 import QtCore, QtGui
import cv2, numpy as np, math, time, logging
import socketio
import uuid
import datetime
from collections import deque

class Display_Manager(QtCore.QThread):
    change_pixmap_signal = QtCore.pyqtSignal(QtGui.QImage)
    metrics_signal = QtCore.pyqtSignal(int, dict)
    connection_status_signal = QtCore.pyqtSignal(str)

    def __init__(self, video_sources, toggles, processors, helpers, stream_flag: int = 0):
        super().__init__()
        self.video_sources = video_sources
        self.toggles = toggles
        self.processors = processors
        self.helpers = helpers
        self._stop = False
        self.stream_flag = stream_flag
        
        # Server communication setup
        self.sio = socketio.Client(reconnection=True, reconnection_attempts=5)
        self.device_id = None
        self.http_device_id = f"transmitter_{str(uuid.uuid4())[:8]}"
        self.device_type = helpers.get('device_config').device_type
        self.server_url = helpers.get('device_config').get_server_url()
        self.setup_socket_handlers()
        
        self._connection_retry_count = 0
        self._max_connection_retries = 5
        # Periodic sensor-only send setup
        self._last_sensor_send = 0.0
        self._sensor_send_interval = 5.0  # seconds between sensor-only sends

    def setup_socket_handlers(self):
        @self.sio.on('connect')
        def on_connect():
            self.connection_status_signal.emit("Connected to server")
            self.self_identify_device()  # New: self-identify on connect

        @self.sio.on('disconnect')
        def on_disconnect():
            self.connection_status_signal.emit("Disconnected from server")

        @self.sio.on('self_identity_ack')
        def on_self_identity_ack(data):
            if data.get('status') == 'success' and data.get('device_id') == self.http_device_id:
                self.device_id = self.http_device_id  # Confirm internal device_id upon ack
                self.connection_status_signal.emit(f"Self-identity acknowledged by server for device: {self.device_id}")
            else:
                self.connection_status_signal.emit(f"Self-identity acknowledgement failed or mismatched for {self.http_device_id}. Server response: {data}")

        @self.sio.on('config_update')
        def on_config_update(config):
            logging.info(f"Received config_update via socket: {config}")
            self.on_config_update(config)

    def connect_to_server(self):
        if hasattr(self.helpers.get('device_config'), 'server_enabled') and not self.helpers.get('device_config').server_enabled:
            self.connection_status_signal.emit("Server connection disabled in config")
            return
            
        try:
            import requests
            try:
                server_check = requests.get(f"{self.server_url}/api/status", timeout=2)
                if server_check.status_code != 200:
                    raise Exception(f"Server returned status code: {server_check.status_code}")
            except requests.exceptions.RequestException as e:
                logging.error(f"Server is not accessible: {e}")
                raise Exception(f"Server at {self.server_url} appears to be down: {e}")
            
            if not self.sio.connected:
                self.sio = socketio.Client(
                    reconnection=True, 
                    reconnection_attempts=10,
                    reconnection_delay=1,
                    reconnection_delay_max=5,
                    engineio_logger=True
                )
                self.setup_socket_handlers()
                
                logging.info(f"Attempting to connect to server at {self.server_url}")
                self.connection_status_signal.emit(f"Connecting to server at {self.server_url}...")
                self.sio.connect(self.server_url)
        except Exception as e:
            logging.error(f"Failed to connect to server: {e}")
            self._connection_retry_count += 1
            if self._connection_retry_count < self._max_connection_retries:
                retry_wait = min(2 * self._connection_retry_count, 10)
                self.connection_status_signal.emit(f"Connection attempt {self._connection_retry_count} failed. Retrying in {retry_wait}s...")
                time.sleep(retry_wait)
                self.connect_to_server()
            else:
                logging.error("Max connection retries reached.")
                self.connection_status_signal.emit(f"Server connection failed after {self._max_connection_retries} attempts - continuing in offline mode")

    def self_identify_device(self):
        if self.sio.connected:
            logging.info(f"Attempting to self-identify as device_id: {self.http_device_id}, type: {self.device_type}")
            self.sio.emit('self_identify', {'device_id': self.http_device_id, 'device_type': self.device_type})
        else:
            logging.warning("Cannot self-identify: Socket not connected.")

    def send_metrics_to_server(self, channel_id: int, metrics: dict):
        device_id_to_send = self.http_device_id

        if not device_id_to_send:
            logging.error("http_device_id is not set. Cannot send metrics.")
            return

        formatted_metrics = {
            'people_count': metrics.get('people_in', 0) + metrics.get('crowd', 0),
            'lab_coat_count': metrics.get('labcoats', 0) + metrics.get('labcoat_entries', 0),
        }
        
        for key, value in metrics.items():
            if key not in ['people_count', 'lab_coat_count']:
                formatted_metrics[key] = value
                
        # Handle tower light metrics, prioritizing tower_processor output
        if 'green_durations_per_id' in metrics and isinstance(metrics['green_durations_per_id'], dict):
            light_timers = metrics['green_durations_per_id']
            sorted_light_items = list(light_timers.items()) 
            
            for i in range(len(sorted_light_items)):
                if i < 4:
                    _lid, duration = sorted_light_items[i]
                    formatted_metrics[f'light_{i+1}_status'] = f"{duration}s"
                else:
                    break
            
            if metrics.get('green_lights_active_count', 0) > 0:
                formatted_metrics['tower_light'] = 'Green'
            elif metrics.get('detected_lights_count', 0) > 0:
                formatted_metrics['tower_light'] = 'Yellow'
            elif metrics.get('detected_lights_count', 0) == 0:
                formatted_metrics['tower_light'] = 'Off'

        elif 'green_duration_s' in metrics:
            formatted_metrics['tower_light'] = 'Green' if metrics.get('green_active', 0) == 1 else 'Red/Yellow'

        if 'illegal_exit' in metrics:
            formatted_metrics['door_status'] = metrics.get('door_status', 'Unknown')
            
        if 'entrance_in' in metrics:
            formatted_metrics['entrance_in'] = metrics.get('entrance_in', 0)
            formatted_metrics['entrance_out'] = metrics.get('entrance_out', 0)
            
        if 'efficiency' in metrics:
            formatted_metrics['efficiency'] = metrics.get('efficiency', 0)

        data = {
            'device_id': device_id_to_send,
            'device_type': self.device_type,
            'channel': channel_id,
            'metrics': formatted_metrics,
            'timestamp': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        }
        logging.debug(f"send_metrics_to_server: current sensor_readings={self.helpers.get('sensor_readings')}")
        data['sensor_data'] = self.helpers.get('sensor_readings', {})
        sensor_err = self.helpers.get('sensor_readings_error')
        if sensor_err:
            data['sensor_status_error'] = sensor_err

        try:
            import requests
            logging.info(f"Sending metrics to server via HTTP: channel {channel_id}, metrics: {formatted_metrics}")
            response = requests.post(f"{self.server_url}/update_metrics", json=data, timeout=2)
            if not response.ok:
                logging.error(f"HTTP metrics send failed: {response.status_code} - {response.text}")
        except Exception as e:
            logging.error(f"Failed to send metrics to server via HTTP: {e}")
            try:
                if self.sio.connected:
                    self.sio.emit('update_metrics', data)
            except Exception:
                logging.error("Fallback socket emit also failed")
                if not self.sio.connected:
                    logging.info("Socket disconnected, attempting to reconnect...")
                    self._connection_retry_count = 0
                    self.connect_to_server()

    def send_anomaly_to_server(self, description: str, anomaly_type: str, details: dict):
        device_id_to_send = self.http_device_id
        channel = details.get('channel', 'unknown')

        data = {
            'device_id': device_id_to_send,
            'channel': channel,
            'device_type': self.device_type,
            'description': description,
            'type': anomaly_type,
            'details': details,
            'timestamp': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        }

        if not details.get('sensor_type') and 'sensor_readings' in self.helpers and isinstance(self.helpers['sensor_readings'], dict) and self.helpers['sensor_readings']:
            data['sensor_data'] = self.helpers['sensor_readings']

        try:
            import requests
            logging.info(f"Sending anomaly to server via HTTP: {description}")
            response = requests.post(f"{self.server_url}/anomaly", json=data, timeout=2)
            if not response.ok:
                logging.error(f"HTTP anomaly send failed: {response.status_code} - {response.text}")
        except Exception as e:
            logging.error(f"Failed to send anomaly via HTTP: {e}")
            try:
                if self.sio.connected:
                    self.sio.emit('anomaly', data)
            except Exception as e2:
                logging.error(f"Fallback socket anomaly emit failed: {e2}")

    def send_sensor_readings(self):
        """Send only sensor data to server if available."""
        readings = self.helpers.get('sensor_readings')
        if isinstance(readings, dict) and readings:
            device_id_to_send = self.http_device_id
            payload = {
                'device_id': device_id_to_send,
                'device_type': self.device_type,
                'channel': 'sensor',
                'timestamp': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                'sensor_data': readings
            }
            try:
                import requests
                logging.info(f"Sending sensor-only data to server: {readings}")
                requests.post(f"{self.server_url}/update_metrics", json=payload, timeout=2)
            except Exception as e:
                logging.error(f"Failed sensor-only HTTP send: {e}")
                if self.sio.connected:
                    try: self.sio.emit('update_metrics', payload)
                    except: pass

    def stop(self):
        self._stop = True
        if self.sio.connected:
            self.sio.disconnect()

    def on_config_update(self, config):
        if 'crowd' in config:
            self.helpers['crowd_threshold'] = config['crowd']['group_size']
            self.helpers['crowd_secs'] = config['crowd']['duration']
        
        if 'lab_coat' in config:
            self.helpers['labcoat_missing_secs'] = config['lab_coat']['missing_duration']
        
        if 'tower_light' in config:
            self.helpers['tower_green_threshold'] = config['tower_light']['green_duration']
        
        if 'detection' in config:
            self.helpers['human_conf'] = config['detection']['human_confidence']
            self.helpers['labcoat_conf'] = config['detection']['labcoat_confidence']
            self.helpers['light_conf'] = config['detection']['light_confidence']
            self.helpers['door_conf'] = config['detection']['door_confidence']

        if 'sensors' in config and isinstance(config['sensors'], dict):
            device_config_obj = self.helpers.get('device_config')
            if device_config_obj:
                if not hasattr(device_config_obj, 'sensor_config') or not isinstance(device_config_obj.sensor_config, dict):
                    logging.info("Initializing 'sensor_config' on device_config object as it was missing or not a dict.")
                    try:
                        device_config_obj.sensor_config = {}
                    except Exception as e:
                        logging.error(f"Could not create 'sensor_config' on device_config: {e}. Sensor configs from server may not apply.")

                if hasattr(device_config_obj, 'sensor_config') and isinstance(device_config_obj.sensor_config, dict):
                    logging.info(f"Updating device_config.sensor_config. Current: {device_config_obj.sensor_config}, With: {config['sensors']}")
                    device_config_obj.sensor_config.update(config['sensors'])
                    logging.info(f"After update, device_config.sensor_config: {device_config_obj.sensor_config}")
                else:
                    logging.error("device_config.sensor_config is not a dictionary after attempted initialization, cannot update with sensor settings from server.")
            else:
                logging.warning("device_config not found in helpers. Cannot update sensor configurations.")

    def update_server_address(self, new_url):
        """Update server URL at runtime and reconnect client."""
        self.server_url = new_url
        self.connection_status_signal.emit(f"Server reconfigured to {new_url}, reconnecting...")
        try:
            if self.sio.connected:
                self.sio.disconnect()
        except Exception:
            pass
        self._connection_retry_count = 0
        self.connect_to_server()

    def run(self):
        self.connect_to_server()

        try:
            import requests
            config_response = requests.get(f"{self.server_url}/api/config", timeout=2)
            if config_response.ok:
                self.on_config_update(config_response.json())
        except Exception as e:
            logging.error(f"Failed to fetch initial configuration: {e}")
        self._last_config_fetch = time.time()
        self._config_refresh_interval = 10.0

        caps = []
        for i, src in enumerate(self.video_sources):
            try:
                if self.stream_flag:
                    cap = cv2.VideoCapture(src["path"], cv2.CAP_FFMPEG)
                    if not cap.isOpened():
                        cap = cv2.VideoCapture(src["path"], cv2.CAP_GSTREAMER)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                else:
                    cap = cv2.VideoCapture(src["path"])
                
                if not cap.isOpened():
                    path = src["path"]
                    if not path.startswith("rtsp://") and not path.startswith("http://") and not path.startswith("https://"):
                        path = path.replace("\\", "/")
                    cap = cv2.VideoCapture(path)
                
                if not cap.isOpened():
                    logging.error(f"Could not open video source {i}: {src['path']}. Please check the path and ensure video codecs are installed.")
                    continue
                
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                caps.append(cap)
                logging.info(f"Successfully opened video source {i}: {src['path']}")
            except Exception as e:
                logging.error(f"Exception while opening video source {i}: {src['path']} - {str(e)}")
        
        if not caps:
            logging.error("No video captures were successfully opened. Check video paths and codecs.")
            self.connection_status_signal.emit("Error: No video sources available")
            return

        if self.stream_flag:
            try:
                fps0 = caps[0].get(cv2.CAP_PROP_FPS)
                if fps0 <= 0 or math.isnan(fps0):
                    fps0 = 30
                    logging.warning(f"Invalid FPS detected, using default value: {fps0}")
            except:
                fps0 = 30
                logging.warning(f"Could not get FPS, using default value: {fps0}")
                
            delay_frames = int(fps0 * 1.0)
            frame_buffers = [deque(maxlen=delay_frames) for _ in caps]
        else:
            frame_buffers = None

        n = len(caps)
        rows = math.ceil(math.sqrt(n))
        cols = math.ceil(n / rows)

        SUB_W, SUB_H = 704, 576

        frame_times = deque(maxlen=30)
        target_fps = 15
        skip_frames = 0
        
        while not self._stop and any(c.isOpened() for c in caps):
            try:
                now_conf = time.time()
                if now_conf - self._last_config_fetch >= self._config_refresh_interval:
                    import requests
                    resp = requests.get(f"{self.server_url}/api/config", timeout=2)
                    if resp.ok:
                        logging.info(f"Received periodic config update via HTTP: {resp.json()}")
                        self.on_config_update(resp.json())
                    self._last_config_fetch = now_conf
            except Exception as e:
                logging.debug(f"Config refresh failed: {e}")

            now = time.time()
            if now - self._last_sensor_send >= self._sensor_send_interval:
                self.send_sensor_readings()
                self._last_sensor_send = now

            frame_start = time.time()
            
            sensor_anomaly_queue = self.helpers.get('sensor_anomaly_data', {}).get('anomaly_event_queue')
            if sensor_anomaly_queue:
                processed_count = 0
                max_to_process_per_cycle = 5
                while sensor_anomaly_queue and processed_count < max_to_process_per_cycle:
                    try:
                        anomaly_event = sensor_anomaly_queue.popleft()
                        
                        description = anomaly_event.get('description', 'Unknown sensor anomaly')
                        event_type = anomaly_event.get('type', 'sensor_event')
                        event_details = anomaly_event.get('details', {})
                        
                        if 'channel' not in event_details:
                             event_details['channel'] = anomaly_event.get('channel', 'unknown_sensor_channel')

                        self.send_anomaly_to_server(description, event_type, event_details)
                        
                        gui_anomaly_message = f"[Sensor] {description}" 
                        if 'anomaly_cb' in self.helpers:
                            self.helpers['anomaly_cb'](gui_anomaly_message)
                        else:
                            logging.warning("anomaly_cb not found in helpers. Sensor GUI anomaly update skipped.")
                        processed_count += 1
                    except IndexError:
                        logging.debug("Sensor anomaly queue processed.")
                        break 
                    except Exception as e:
                        logging.error(f"Error processing sensor anomaly from queue: {e}", exc_info=True)
                        break 

            mosaic = np.zeros((rows*SUB_H, cols*SUB_W, 3), dtype=np.uint8)

            for idx, cap in enumerate(caps):
                if skip_frames > 0:
                    if self.stream_flag:
                        ok, _ = cap.read()
                        if ok and frame_buffers[idx]:
                            frame_buffers[idx].append(None)
                        continue
                    else:
                        ok, _ = cap.read()
                        continue

                if self.stream_flag:
                    ok, grabbed = cap.read()
                    if ok:
                        frame_buffers[idx].append(grabbed)
                    if len(frame_buffers[idx]) < frame_buffers[idx].maxlen:
                        frame = np.zeros((SUB_H, SUB_W, 3), dtype=np.uint8)
                    else:
                        frame = frame_buffers[idx][0]
                else:
                    ok, raw = cap.read()
                    if not ok:
                        raw = np.zeros((SUB_H, SUB_W, 3), dtype=np.uint8)
                    frame = raw

                ch_type = self.video_sources[idx]["type"]
                channel_toggles = self.toggles.get(idx, {})
                proc_list = self.processors.get(ch_type, [])
                overlay_native = frame.copy()
                merged_metrics = {}

                for proc in proc_list:
                    try:
                        overlay_native, metrics = proc(overlay_native,
                                                     self.helpers,
                                                     idx,
                                                     channel_toggles)
                        merged_metrics.update(metrics)
                        if metrics.get('anomaly_msg'):
                            if 'anomaly_cb' in self.helpers:
                                self.helpers['anomaly_cb'](metrics['anomaly_msg'])
                            else:
                                logging.warning("anomaly_cb not found in helpers. Processor GUI anomaly update skipped.")
                            
                            details = metrics.get('anomaly_details', {})
                            details['channel'] = idx
                            self.send_anomaly_to_server(
                                metrics['anomaly_msg'],
                                metrics.get('anomaly_type', 'general_processor_anomaly'),
                                details
                            )
                    except Exception as e:
                        logging.exception(f"Processor error in channel {idx}: {e}")

                overlay = cv2.resize(overlay_native, (SUB_W, SUB_H))
                if merged_metrics:
                    self.metrics_signal.emit(idx, merged_metrics)
                    self.send_metrics_to_server(idx, merged_metrics)

                r, c = divmod(idx, cols)
                mosaic[r*SUB_H:(r+1)*SUB_H, c*SUB_W:(c+1)*SUB_W] = overlay

            frame_time = time.time() - frame_start
            frame_times.append(frame_time)
            avg_frame_time = sum(frame_times) / len(frame_times)
            current_fps = 1 / avg_frame_time if avg_frame_time > 0 else 0

            if current_fps < target_fps * 0.8:
                skip_frames = min(skip_frames + 1, 2)
            elif current_fps > target_fps * 1.2:
                skip_frames = max(skip_frames - 1, 0)

            if skip_frames > 0:
                skip_frames -= 1

            rgb = cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qt_image = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
            self.change_pixmap_signal.emit(qt_image)

        for cap in caps:
            cap.release()

