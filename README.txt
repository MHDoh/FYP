Overview
This project is a vision‑and‑sensor based monitoring system with a desktop GUI client (“transmitter”) and a Flask + Socket.IO web server. It performs real‑time video analytics (people detection, lab‑coat compliance, entrance/exit monitoring, tower light status) and fuses environmental sensor readings for anomaly detection. The system is designed to run on Jetson devices or standard computers and supports both local GUI monitoring and centralized web dashboards.

Core Components

Transmitter (Client)
Entry point: main.py
Loads ML models (YOLO and a door classifier), initializes device configuration, and starts the GUI.
Runs a sensor thread that parses UART data (CO₂, humidity, temperature, acceleration, pressure).
Performs consecutive‑anomaly checks and forwards events to the server.
Supports headless mode to send periodic sensor data without GUI.
GUI Application
UI layer and Socket.IO client: main_window.py
Displays video feeds, metrics, toggles, and anomaly logs.
Connects to server for configuration updates and metric reporting.
Settings dialog and device info update handling.
Video Analytics Pipeline
Processors and algorithms: processors.py
Key analytics:
Entrance monitoring: turnstile flow, red‑light violations, lab‑coat compliance.
Exit monitoring: door open + human + object logic for illegal exits.
Tower light monitoring and crowd/people density metrics.
Uses YOLO for detection and OpenCV for ROI/motion analysis.
Anomaly Manager
Lightweight FIFO notifier for GUI: anomaly_manager.py
Thread‑safe signal emitter to update anomaly panel in the UI.
Server (Dashboard + API)
Backend: server.py
Flask + Socket.IO, user login, dashboard, configuration management, and anomaly broadcasting.
Stores metrics, heartbeats, sensor data, and anomaly history.
Emits live updates to web UI and connected clients.
Device & Environment Configuration
Device config: device_config.py
Handles model paths, server address, Jetson detection, and sensor UART setup.
Default configuration module: config.py
Key Features

Real‑time people detection, lab‑coat compliance checks, and illegal entry detection.
Exit monitoring based on door state, human presence, and object detection.
Tower light monitoring with duration‑based anomaly triggers.
Integrated sensor pipeline (CO₂, humidity, temperature, pressure, acceleration).
Configurable thresholds with server‑side updates and live client reconfiguration.
Multi‑channel support with per‑channel toggles and global controls.
Web dashboard with live metrics, anomaly feed, and config page.
Persistent logging for transmitter and server.
Data Flow Summary

Transmitter captures video and runs analytics.
Metrics and anomalies are sent to the server via Socket.IO or HTTP.
Sensor data is read from UART, validated, aggregated, and sent periodically.
Server validates anomalies, stores metrics, and broadcasts updates to web UI and connected clients.
Models and Assets

ML models in models:
YOLO human detection: yolo11n.pt
Light detection: best_light.pt
Lab‑coat detection: labcoat_with_mall.pt
Object exit detection: object_exit.pt
Door state classifier: door_model.pt
Dependencies

Python packages listed in requirements.txt, including PyTorch, Ultralytics YOLO, OpenCV, PyQt5, and Flask‑CORS.
How to Run (high level)

Start the server (web dashboard and API): run server.py
Start the transmitter client (GUI + analytics): run main.py
