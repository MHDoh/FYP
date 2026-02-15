"""Device configuration and platform-specific settings."""
import platform
import os
from pathlib import Path

class DeviceConfig:
    def __init__(self):
        self.platform = platform.system()
        # Auto-detect Jetson but allow manual override
        self._auto_detected_jetson = self._check_if_jetson()
        self.is_jetson = self._auto_detected_jetson
        self.device_type = "jetson" if self.is_jetson else "computer"
        
        # Base paths
        self.base_dir = Path(__file__).resolve().parent.parent
        self.models_dir = self.base_dir / "models"
        self.logs_dir = self.base_dir / "logs"

        # Model filenames
        self.door_model_filename = "door_model.pt"
        self.yolo_human_model_filename = "yolo11n.pt"
        self.light_model_filename = "best_light.pt"
        self.labcoat_model_filename = "labcoat_with_mall.pt"
        self.object_model_filename = "object_exit.pt"  # Added for the new object exit model

        # Server configuration
        self.server_host = "127.0.0.1"  # Update this to your server's actual IP address
        self.server_port = 5000
        self.server_enabled = True  # Enable server connection
        
        # Sensor configuration (Jetson only)
        self.sensor_config = {
            "enabled": self.is_jetson,
            "uart_port": "/dev/ttyTHS1",
            "baud_rate": 115200,
            "timeout": 1
        }
        
        # Override sensor port and baud from environment if provided
        self.sensor_config['uart_port'] = os.environ.get('SENSOR_PORT', self.sensor_config.get('uart_port'))
        self.sensor_config['baud_rate'] = int(os.environ.get('SENSOR_BAUD', self.sensor_config.get('baud_rate', 115200)))

    def _check_if_jetson(self):
        """Check if running on Jetson Nano Orin."""
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read()
                return 'Jetson' in model
        except:
            return False

    def get_device_info(self):
        """Get device-specific information."""
        return {
            "type": self.device_type,
            "platform": self.platform,
            "has_sensors": self.is_jetson,
            "models_path": str(self.models_dir),
            "logs_path": str(self.logs_dir)
        }

    def get_server_url(self):
        """Get the complete server URL."""
        return f"http://{self.server_host}:{self.server_port}"
        
    def set_server_address(self, host, port=5000):
        """Manually set the server IP address and port."""
        self.server_host = host
        self.server_port = port
        
    def set_device_type(self, device_type):
        """Manually set the device type ('jetson' or 'computer')."""
        if device_type.lower() in ["jetson", "computer"]:
            self.device_type = device_type.lower()
            self.is_jetson = (self.device_type == "jetson")
            # Update sensor configuration based on device type
            self.sensor_config["enabled"] = self.is_jetson
            return True
        return False
        
    def get_auto_detected_device_type(self):
        """Return the auto-detected device type."""
        return "jetson" if self._auto_detected_jetson else "computer"