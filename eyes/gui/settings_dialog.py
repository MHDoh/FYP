"""Settings dialog for device configuration."""
from PyQt5 import QtWidgets, QtCore, QtGui

class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, device_config, parent=None):
        super().__init__(parent)
        self.device_config = device_config
        self.setWindowTitle("Settings")
        self.setMinimumWidth(400)
        self.setup_ui()
        
    def setup_ui(self):
        self.setStyleSheet("""
            QDialog { background-color: #1E1E2E; color: #CDD6F4; }
            QLabel { color: #CDD6F4; }
            QGroupBox { color: #CDD6F4; border: 1px solid #45475A; border-radius: 4px; margin-top: 10px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { background-color: #7aa2f7; color: #1a1b26; border: none; border-radius: 3px; padding: 8px 15px; font-weight: bold; }
            QPushButton:hover { background-color: #89b4fa; }
            QPushButton:pressed { background-color: #6c7efa; }
            QLineEdit { background-color: #313244; color: #CDD6F4; border: 1px solid #45475A; border-radius: 3px; padding: 4px; }
            QComboBox { background-color: #313244; color: #CDD6F4; border: 1px solid #45475A; border-radius: 3px; padding: 4px; }
            QComboBox::drop-down { border: 0px; }
            QComboBox::down-arrow { image: url(dropdown-arrow.png); width: 12px; height: 12px; }
            QComboBox QAbstractItemView { background-color: #313244; color: #CDD6F4; selection-background-color: #45475A; }
        """)
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # Server settings group
        server_group = QtWidgets.QGroupBox("Server Settings")
        server_layout = QtWidgets.QFormLayout()
        
        self.server_ip_edit = QtWidgets.QLineEdit(self.device_config.server_host)
        server_layout.addRow("Server IP:", self.server_ip_edit)
        
        self.server_port_edit = QtWidgets.QLineEdit(str(self.device_config.server_port))
        self.server_port_edit.setValidator(QtGui.QIntValidator(1, 65535))
        server_layout.addRow("Server Port:", self.server_port_edit)
        
        server_group.setLayout(server_layout)
        layout.addWidget(server_group)
        
        # Device settings group
        device_group = QtWidgets.QGroupBox("Device Settings")
        device_layout = QtWidgets.QFormLayout()
        
        self.device_type_combo = QtWidgets.QComboBox()
        self.device_type_combo.addItems(["computer", "jetson"])
        current_index = 0 if self.device_config.device_type == "computer" else 1
        self.device_type_combo.setCurrentIndex(current_index)
        
        device_layout.addRow("Device Type:", self.device_type_combo)
        
        auto_detected = self.device_config.get_auto_detected_device_type()
        auto_detected_label = QtWidgets.QLabel(f"Auto-detected as: {auto_detected}")
        auto_detected_label.setStyleSheet("font-style: italic; color: #94e2d5;")
        device_layout.addRow("", auto_detected_label)
        
        device_group.setLayout(device_layout)
        layout.addWidget(device_group)
        
        # Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept_changes)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
    def accept_changes(self):
        # Save server settings
        server_ip = self.server_ip_edit.text().strip()
        try:
            server_port = int(self.server_port_edit.text())
            self.device_config.set_server_address(server_ip, server_port)
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Port", "Please enter a valid port number (1-65535)."
            )
            return
        
        # Save device type
        device_type = self.device_type_combo.currentText()
        self.device_config.set_device_type(device_type)
        
        self.accept()