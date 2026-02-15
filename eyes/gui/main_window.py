# Renaming gui.py to main_window.py for consistency with imports.
# Original content of gui.py will be here.
# main_window.py
from PyQt5 import QtWidgets, QtCore, QtGui
import logging, torch, os, sys, requests, socketio
from pathlib import Path
from ..core.display_manager import Display_Manager
from ..core.anomaly_manager import AnomalyManager
from ..core import processors
import cv2, numpy as np
from eyes.gui.settings_dialog import SettingsDialog

# Global dictionary to hold shared data and models, populated from main.py
helpers = {}
# Global lock for synchronizing access to shared data, especially sensor readings
sensor_data_lock = None

# Function to check server status
def check_server_status():
    if not helpers or 'device_config' not in helpers:
        logging.error("Device config not available for server status check.")
        return False
    server_host = helpers['device_config'].server_host
    server_port = helpers['device_config'].server_port
    try:
        response = requests.get(f"http://{server_host}:{server_port}/api/status", timeout=2)
        return response.status_code == 200 and response.json().get("status") == "active"
    except requests.exceptions.RequestException as e:
        logging.error(f"Server is not accessible: {e}")
        return False

# Function to get initial configuration from server
def get_initial_config():
    if not helpers or 'device_config' not in helpers:
        logging.error("Device config not available for initial config fetch.")
        return None
    server_host = helpers['device_config'].server_host
    server_port = helpers['device_config'].server_port
    try:
        response = requests.get(f"http://{server_host}:{server_port}/api/config", timeout=5)
        if response.status_code == 200:
            logging.info(f"Successfully fetched initial configuration: {response.json()}")
            return response.json()
        else:
            logging.error(f"Failed to fetch initial configuration, status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch initial configuration: {e}")
        return None

# Function to send metrics to the server via HTTP POST
def send_sensor_data_to_server(sensor_data):
    """Sends sensor data to the server if available."""
    if not helpers or 'device_config' not in helpers:
        logging.error("Device config not available for sending sensor data.")
        return

    server_host = helpers['device_config'].server_host
    server_port = helpers['device_config'].server_port
    api_url = f"http://{server_host}:{server_port}/update_sensor_data"

    payload = {
        "device_id": helpers['device_config'].device_id,
        "timestamp": get_utc_timestamp_gui_side(),
        "sensors": sensor_data
    }

    try:
        response = requests.post(api_url, json=payload, timeout=5)
        response.raise_for_status()
        logging.info(f"Sensor data sent successfully: {payload}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to send sensor data to server: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while sending sensor data: {e}")

def send_sensor_data_periodically():
    """Periodically sends sensor data to the server."""
    if sensor_data_lock is None:
        logging.error("Sensor data lock not initialized. Cannot send sensor data.")
        return

    with sensor_data_lock:
        current_sensor_readings = helpers.get('sensor_readings')
        sensor_error = helpers.get('sensor_readings_error')

    if current_sensor_readings is not None:
        logging.info(f"Sending sensor-only data to server: {current_sensor_readings}")
        send_sensor_data_to_server(current_sensor_readings)
    elif sensor_error:
        logging.warning(f"Sensor readings might be unavailable due to error: {sensor_error}. Sending empty sensor data or last known if applicable.")
        send_sensor_data_to_server({})
    else:
        logging.warning("Sensor readings are None, and no specific sensor error is set. Sending empty sensor data.")
        send_sensor_data_to_server({})

# Initialize Socket.IO client
sio = socketio.Client(logger=True, engineio_logger=True)

# Define event handlers for Socket.IO
@sio.event
def connect():
    logging.info("Socket.IO: Successfully connected to server.")
    if helpers and 'device_config' in helpers:
        sio.emit('register_transmitter', {'device_id': helpers['device_config'].device_id})
        logging.info(f"Socket.IO: Sent registration for device_id: {helpers['device_config'].device_id}")
    else:
        logging.warning("Socket.IO: Device config not available at connection time for registration.")

@sio.event
def disconnect():
    logging.info("Socket.IO: Disconnected from server.")

@sio.event
def config_update(data):
    logging.info(f"Received config_update via socket: {data}")
    if helpers and 'device_config' in helpers:
        try:
            if 'crowd' in data:
                helpers['device_config'].crowd_config.update(data['crowd'])
            if 'lab_coat' in data:
                helpers['device_config'].lab_coat_config.update(data['lab_coat'])
            if 'tower_light' in data:
                helpers['device_config'].tower_light_config.update(data['tower_light'])
            if 'detection' in data:
                helpers['device_config'].detection_thresholds.update(data['detection'])
            if 'sensors' in data:
                server_sensor_config = data.get('sensors', {})
                if 'window_size' in server_sensor_config:
                    helpers['device_config'].sensor_config['window_size'] = server_sensor_config['window_size']
                for sensor_key, thresholds in server_sensor_config.items():
                    if sensor_key == 'window_size': continue
                    if isinstance(thresholds, dict):
                        # This is where you need to ensure the 'acceleration' thresholds are correctly updated
                        if sensor_key == 'acceleration':
                            if 'x_threshold' in thresholds and 'y_threshold' in thresholds and 'z_threshold' in thresholds:
                                 helpers['device_config'].sensor_config.setdefault('acceleration', {}).update(thresholds)
                            else:
                                 # Handle old format or log an error if necessary
                                 logging.warning(f"Received incomplete acceleration config: {thresholds}")
                        elif sensor_key in helpers['device_config'].sensor_config: # For other sensors like co2, temp etc.
                            helpers['device_config'].sensor_config[sensor_key].update(thresholds)
                        else:
                            helpers['device_config'].sensor_config[sensor_key] = thresholds
            logging.info("Device configuration updated from server.")
            helpers['device_config'].save_to_file()
            logging.info("Updated device configuration saved to file.")
        except Exception as e:
            logging.error(f"Error updating device_config from socket data: {e}")
            logging.debug(f"Data causing error: {data}")

def connect_to_socketio_server():
    if not helpers or 'device_config' not in helpers:
        logging.error("Device config not available for Socket.IO connection.")
        return
    server_host = helpers['device_config'].server_host
    server_port = helpers['device_config'].server_port
    try:
        logging.info(f"Attempting to connect to server at http://{server_host}:{server_port}")
        sio.connect(f"http://{server_host}:{server_port}", transports=['websocket', 'polling'])
    except socketio.exceptions.ConnectionError as e:
        logging.error(f"Socket.IO connection failed: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during Socket.IO connection: {e}")

def start_gui(video_sources, shared_helpers, stream_flag):
    global helpers, sensor_data_lock
    helpers = shared_helpers
    sensor_data_lock = shared_helpers.get('sensor_data_lock')

    if not sensor_data_lock:
        logging.critical("Sensor data lock not found in shared_helpers. Sensor data synchronization will fail.")

    stop_all = False
    device_config = helpers.get('device_config')

    global_toggles = {}
    for i, src in enumerate(video_sources):
        if src["type"] == "standard":
            global_toggles[i] = {'density': True, 'labcoat': True}
        elif src["type"] == "entrance":
            global_toggles[i] = {'density': True, 'yolo': True, 'labcoat': True, 'entrance': True}
        elif src["type"] == "tower":
            global_toggles[i] = {'light': True, 'density': True, 'labcoat': True}
        elif src["type"] == "exit":
            global_toggles[i] = {'density': True, 'yolo': True, 'exit': True, 'labcoat': True}

    package_root = Path(__file__).resolve().parent.parent
    LOG_DIR = package_root / "logs"
    os.makedirs(LOG_DIR, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_DIR / "transmitter.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    processors_map = {
        "entrance": [ processors.density_processor,
                      processors.labcoat_processor,
                      processors.entrance_processor ],

        "exit":     [ processors.density_processor,
                      processors.labcoat_processor,
                      processors.exit_processor ],

        "tower":    [ processors.density_processor,
                      processors.labcoat_processor,
                      processors.tower_processor ],

        "standard": [ processors.density_processor,
                      processors.labcoat_processor ],
    }

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.device_config = device_config
            self.setWindowTitle(f"Advanced Video Monitoring System - {device_config.device_type.upper()}")
            self.resize(1280, 800)
            self.setup_ui()

        def setup_ui(self):
            self.setStyleSheet("""
                QMainWindow { background-color: #1E1E2E; color: #CDD6F4; }
                QLabel { color: #CDD6F4; }
                QGroupBox { color: #CDD6F4; border: 1px solid #45475A; border-radius: 4px; margin-top: 10px; font-weight: bold; }
                QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
                QPushButton { background-color: #7aa2f7; color: #1a1b26; border: none; border-radius: 3px; padding: 8px 15px; font-weight: bold; }
                QPushButton:hover { background-color: #89b4fa; }
                QPushButton:pressed { background-color: #6c7efa; }
                QCheckBox { color: #CDD6F4; }
            """)

            central_widget = QtWidgets.QWidget()
            self.setCentralWidget(central_widget)
            main_layout = QtWidgets.QHBoxLayout(central_widget)
            main_layout.setContentsMargins(10, 10, 10, 10)
            main_layout.setSpacing(10)

            left_panel = QtWidgets.QWidget()
            left_panel.setFixedWidth(300)
            left_layout = QtWidgets.QVBoxLayout(left_panel)

            device_group = QtWidgets.QGroupBox("Device Information")
            device_layout = QtWidgets.QVBoxLayout()
            self.device_type_label = QtWidgets.QLabel(f"Type: {device_config.device_type}")
            self.platform_label = QtWidgets.QLabel(f"Platform: {device_config.platform}")
            self.server_status_label = QtWidgets.QLabel("Server: Disconnected")
            self.device_id_label = QtWidgets.QLabel("Device ID: None")
            
            device_layout.addWidget(self.device_type_label)
            device_layout.addWidget(self.platform_label)
            device_layout.addWidget(self.server_status_label)
            device_layout.addWidget(self.device_id_label)
            device_group.setLayout(device_layout)
            left_layout.addWidget(device_group)

            self.channel_widgets = {}
            for ch, src in enumerate(video_sources):
                source_type = src["type"]
                if ch not in global_toggles:
                    continue

                group_box = QtWidgets.QGroupBox(f"Channel {ch} ({source_type})")
                ch_layout = QtWidgets.QVBoxLayout()
                source_path = os.path.basename(src["path"])
                path_label = QtWidgets.QLabel(f"Source: {source_path}")
                path_label.setWordWrap(True)
                ch_layout.addWidget(path_label)
                controls = QtWidgets.QGridLayout()

                chk_density = QtWidgets.QCheckBox("Human-Density")
                chk_density.setChecked(global_toggles[ch].get('density', False))
                controls.addWidget(chk_density, 0, 0)

                chk_labcoat = QtWidgets.QCheckBox("Lab Coat Detection")
                chk_labcoat.setChecked(global_toggles[ch].get('labcoat', False))
                controls.addWidget(chk_labcoat, 1, 0)

                widget_dict = {'density': chk_density, 'labcoat': chk_labcoat}

                if source_type == "tower":
                    chk_light = QtWidgets.QCheckBox("Light Detection")
                    chk_light.setChecked(global_toggles[ch].get('light', False))
                    controls.addWidget(chk_light, 2, 0)
                    widget_dict['light'] = chk_light
                    chk_light.stateChanged.connect(lambda state, ch=ch: self.update_toggle(ch, 'light', state))

                if source_type == "entrance":
                    chk_entrance = QtWidgets.QCheckBox("Entrance Monitoring")
                    chk_entrance.setChecked(global_toggles[ch].get('entrance', False))
                    controls.addWidget(chk_entrance, 2, 0)
                    widget_dict['entrance'] = chk_entrance
                    chk_entrance.stateChanged.connect(lambda state, ch=ch: self.update_toggle(ch, 'entrance', state))

                if source_type == "exit":
                    chk_exit = QtWidgets.QCheckBox("Exit Monitoring")
                    chk_exit.setChecked(global_toggles[ch].get('exit', False))
                    controls.addWidget(chk_exit, 2, 0)
                    widget_dict['exit'] = chk_exit
                    chk_exit.stateChanged.connect(lambda state, ch=ch: self.update_toggle(ch, 'exit', state))
                    chk_labcoat.setToolTip("Enable lab coat compliance detection")

                if source_type in ("entrance", "exit"):
                    chk_yolo = QtWidgets.QCheckBox("YOLO boxes")
                    chk_yolo.setChecked(global_toggles[ch].get('yolo', True))
                    controls.addWidget(chk_yolo, 0, 1)
                    widget_dict['yolo'] = chk_yolo
                    chk_yolo.stateChanged.connect(lambda s, ch=ch: self.update_toggle(ch,'yolo',s))

                chk_density.stateChanged.connect(lambda s, ch=ch: self.update_toggle(ch,'density',s))
                chk_labcoat.stateChanged.connect(lambda state, ch=ch: self.update_toggle(ch, 'labcoat', state))

                self.channel_widgets[ch] = widget_dict
                ch_layout.addLayout(controls)
                group_box.setLayout(ch_layout)
                left_layout.addWidget(group_box)

            global_group = QtWidgets.QGroupBox("Global Controls")
            global_layout = QtWidgets.QVBoxLayout()
            buttons_layout = QtWidgets.QHBoxLayout()
            all_on_btn = QtWidgets.QPushButton("All On")
            all_on_btn.clicked.connect(lambda: self.toggle_all(True))
            all_on_btn.setToolTip("Enable all detection features for all channels")
            all_off_btn = QtWidgets.QPushButton("All Off")
            all_off_btn.clicked.connect(lambda: self.toggle_all(False))
            all_off_btn.setToolTip("Disable all detection features for all channels")
            buttons_layout.addWidget(all_on_btn)
            buttons_layout.addWidget(all_off_btn)
            global_layout.addLayout(buttons_layout)
            
            settings_button = QtWidgets.QPushButton("Settings")
            settings_button.clicked.connect(self.open_settings)
            settings_button.setToolTip("Configure server connection and device settings")
            global_layout.addWidget(settings_button)
            
            global_group.setLayout(global_layout)
            left_layout.addWidget(global_group)

            self.anom_mgr = AnomalyManager()

            anom_group = QtWidgets.QGroupBox("Anomaly Messages")
            anom_layout = QtWidgets.QVBoxLayout()
            self.anom_list = QtWidgets.QListWidget()
            self.anom_list.setFixedHeight(70)
            self.anom_list.setStyleSheet("""
                QListWidget { background-color:#1a1b26; border:1px solid #45475A;
                            color:#f2f2f2; font-size:12px; }
                QListWidget::item { 
                    padding: 2px;
                    border-bottom: 1px solid #2d2e3b;
                }
            """)
            anom_layout.addWidget(self.anom_list)
            
            self.clear_log_button = QtWidgets.QPushButton("Clear Log")
            self.clear_log_button.clicked.connect(self.clear_anomaly_log)
            anom_layout.addWidget(self.clear_log_button)
            
            anom_group.setLayout(anom_layout)
            anom_group.setFixedHeight(140)
            left_layout.addWidget(anom_group)
            def _on_new_anomaly(entry:str):
                item = QtWidgets.QListWidgetItem(entry)
                if "Sensor anomaly" in entry:
                    item.setForeground(QtGui.QColor(255, 165, 0))
                elif "Critical" in entry:
                    item.setForeground(QtGui.QColor(255, 0, 0))
                self.anom_list.insertItem(0, item)
                while self.anom_list.count() > 4:
                    self.anom_list.takeItem(self.anom_list.count() - 1)
            
            self.anom_mgr.anomaly_added.connect(_on_new_anomaly)

            left_layout.addStretch()
            self.btn_stop = QtWidgets.QPushButton("Stop Processing")
            self.btn_stop.setStyleSheet("background-color: #F38BA8; color: #1E1B2E;")
            self.btn_stop.clicked.connect(self.stop_processing)
            left_layout.addWidget(self.btn_stop)

            legend_label = QtWidgets.QLabel()
            legend_label.setFixedHeight(70)
            legend_label.setFixedWidth(left_panel.width() - 20)
            legend_img = self.create_density_legend_image(legend_label.width(), 70)
            legend_pix = QtGui.QPixmap.fromImage(legend_img)
            legend_label.setPixmap(legend_pix)
            left_layout.addWidget(legend_label)

            self.video_label = QtWidgets.QLabel("Video Feed")
            self.video_label.setAlignment(QtCore.Qt.AlignCenter)
            self.video_label.setStyleSheet("background-color: #11111B; border-radius: 5px;")

            main_layout.addWidget(left_panel)
            main_layout.addWidget(self.video_label, 1)

            for ch in self.channel_widgets:
                for key, widget in self.channel_widgets[ch].items():
                    if key == 'density':
                        widget.setToolTip("Enable human density detection and crowd analysis")
                    elif key == 'labcoat':
                        widget.setToolTip("Enable lab coat compliance detection")
                    elif key == 'light':
                        widget.setToolTip("Enable tower light status monitoring")
                    elif key == 'entrance':
                        widget.setToolTip("Enable entrance monitoring for access control")
                    elif key == 'exit':
                        widget.setToolTip("Enable exit monitoring for access control")
                    elif key == 'yolo':
                        widget.setToolTip("Show YOLO detection boxes for debugging")

        def update_connection_status(self, status: str):
            self.server_status_label.setText(f"Server: {status}")
            if "Registered as device" in status:
                device_id = status.split(": ")[1]
                self.device_id_label.setText(f"Device ID: {device_id}")

        def update_toggle(self, channel, feature, state):
            global_toggles[channel][feature] = (state == QtCore.Qt.Checked)

        def toggle_all(self, enabled):
            for ch in global_toggles:
                for key in global_toggles[ch]:
                    global_toggles[ch][key] = enabled
                    if key in self.channel_widgets[ch]:
                        self.channel_widgets[ch][key].setChecked(enabled)

        def stop_processing(self):
            nonlocal stop_all
            stop_all = True
            self.close()

        def set_image(self, qt_image):
            self.video_label.setPixmap(QtGui.QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

        def create_density_legend_image(self, width, height):
            DENSITY_COLORS = [
                ((65, 105, 225), "1 "),
                ((50, 205, 50),  "2 "),
                ((255, 191, 0),  "3 "),
                ((255, 69, 0),   "4 "),
                ((220, 20, 60),  "4+ ")
            ]

            img = np.full((height, width, 3), 40, dtype=np.uint8)
            cv2.rectangle(img, (0, 0), (width - 1, height - 1), (60, 60, 80), 1)

            cv2.putText(img, "Density (# of people)", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            box_width = 25
            spacing   = 10
            x = 10
            y = 40

            for color, label in DENSITY_COLORS:
                cv2.rectangle(img, (x, y), (x + box_width, y + box_width), color, -1)
                cv2.putText(img, label, (x + box_width + 5, y + box_width - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                x += box_width + 30

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            return QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        
        def clear_anomaly_log(self):
            self.anom_list.clear()

        def open_settings(self):
            dialog = SettingsDialog(self.device_config, parent=self)
            if dialog.exec_():
                if hasattr(self, 'device_info_label'):
                    self.device_info_label.setText(f"Device: {self.device_config.get_device_type()} | Server: {self.device_config.get_server_url()}")
                
                if hasattr(self, 'vthread'):
                    self.vthread.update_server_address(self.device_config.get_server_url())

        def update_device_info(self):
            from eyes.config import DeviceConfig
            device_config = DeviceConfig()
            device_type = device_config.get_device_type()
            server_address = device_config.get_server_url()
            
            device_info = f"Device Type: {device_type}\nServer: {server_address}"
            if hasattr(self, 'device_info_label'):
                self.device_info_label.setText(device_info)
    

    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    helpers["anomaly_cb"] = main_window.anom_mgr.add
    main_window.show()

    if check_server_status():
        logging.info("Server is active. Proceeding with Socket.IO connection and config fetch.")
        connect_to_socketio_server()

        if not sio.connected:
            logging.info("Socket.IO not connected, attempting to fetch initial config via HTTP.")
            initial_config = get_initial_config()
            if initial_config and helpers and 'device_config' in helpers:
                try:
                    if 'crowd' in initial_config:
                        helpers['device_config'].crowd_config.update(initial_config['crowd'])
                    if 'lab_coat' in initial_config:
                        helpers['device_config'].lab_coat_config.update(initial_config['lab_coat'])
                    if 'tower_light' in initial_config:
                        helpers['device_config'].tower_light_config.update(initial_config['tower_light'])
                    if 'detection' in initial_config:
                        helpers['device_config'].detection_thresholds.update(initial_config['detection'])
                    if 'sensors' in initial_config:
                        server_sensor_config = initial_config.get('sensors', {})
                        if 'window_size' in server_sensor_config:
                             helpers['device_config'].sensor_config['window_size'] = server_sensor_config['window_size']
                        for sensor_key, thresholds in server_sensor_config.items():
                            if sensor_key == 'window_size': continue
                            if isinstance(thresholds, dict):
                                if sensor_key in helpers['device_config'].sensor_thresholds:
                                    if 'min' in thresholds:
                                        helpers['device_config'].sensor_thresholds[sensor_key]['min'] = thresholds['min']
                                    if 'max' in thresholds:
                                        helpers['device_config'].sensor_thresholds[sensor_key]['max'] = thresholds['max']
                                    if 'rate_change' in thresholds:
                                        helpers['device_config'].sensor_thresholds[sensor_key]['rate_change'] = thresholds['rate_change']
                                    if 'threshold' in thresholds:
                                        helpers['device_config'].sensor_thresholds[sensor_key]['threshold'] = thresholds['threshold']
                                else:
                                    logging.warning(f"Received threshold config for unknown sensor '{sensor_key}' in initial_config.")
                    
                    logging.info("Device configuration updated from initial HTTP fetch.")
                    helpers['device_config'].save_to_file()
                except Exception as e:
                    logging.error(f"Error updating device_config from initial HTTP config: {e}")
            elif not initial_config:
                logging.warning("Failed to fetch initial config via HTTP or no device_config in helpers.")
    else:
        logging.error("Server is not active. Will rely on local config and try to connect later.")

    vthread = Display_Manager(
        video_sources = video_sources,
        toggles       = global_toggles,
        processors    = processors_map,
        helpers       = helpers,
        stream_flag   = stream_flag,
    )

    vthread.change_pixmap_signal.connect(main_window.set_image)
    vthread.connection_status_signal.connect(main_window.update_connection_status)
    helpers['vthread'] = vthread
    vthread.start()
    main_window.vthread = vthread

    app.exec_()
    vthread.stop()
    vthread.wait()
