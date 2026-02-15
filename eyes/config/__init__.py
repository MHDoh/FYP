"""Configuration package for Eyes monitoring system."""
from .device_config import DeviceConfig

# Stream Settings
STREAM = 1

# Model Filenames (relative to the eyes package root)
DOOR_MODEL_FILENAME = "models/door_model.pt"
YOLO_HUMAN_MODEL_FILENAME = "models/yolo11n.pt"
LIGHT_MODEL_FILENAME = "models/best_light.pt"
LABCOAT_MODEL_FILENAME = "models/labcoat_with_mall.pt"
OBJECT_MODEL_FILENAME = "models/object_exit.pt" # Added for the new object exit model

# Video Sources
STANDARD_VIDEO_FILES = []
TOWER_VIDEO_FILES = []
ENTRANCE_VIDEO_FILES = [r"C:\Users\mohdh\OneDrive\Desktop\all_channel_mp_firsts\NVR_ch6_20250224_00.mp4"]
EXIT_VIDEO_FILES = []

# RTSP Stream URLs
STANDARD_RTSP_SOURCES = []
TOWER_RTSP_SOURCES = [
  "rtsp://admin:12345abc@10.50.237.20:554/cam/realmonitor?channel=1&subtype=1"
]
ENTRANCE_RTSP_SOURCES = [ ]
EXIT_RTSP_SOURCES = []

# Default configuration
HELPERS_STATIC_CONFIG = {
    # YOLO PARAMS
    "human_conf": 0.4,
    "human_iou": 0.1,
    "human_classes": [0],
    "human_agnostic": False,

    # DETECTION PARAMS
    "door_conf": 0.6,
    "light_conf": 0.4,
    "labcoat_conf": 0.8,

    # ANOMALY PARAMETERS
    "labcoat_missing_secs": 3,
    "crowd_threshold": 4,
    "crowd_secs": 5,
    "tower_green_threshold": 30,

    # Tower light color mapping
    "color_map": {
        0: "Red",
        1: "Yellow",
        2: "Green",
        3: "Green-Red",
        4: "Green-Yellow",
        5: "Red-Green-Yellow"
    },

    # ROIs and static configs
    "turnstile_rois_stream_mode": [(350,89,450,327), (195,89,295,327)],
    "light_rois_stream_mode": [(300,248,360,258), (145,251,205,261)],
    "turnstile_rois_files_mode": [(350,107,450,345), (195,107,295,345)],
    "light_rois_files_mode": [(300,266,360,276), (145,269,205,279)],
    "door_roi": [422,  37, 535, 325],
    "expanded_roi": [272,   7, 605, 415]
}

VIDEO_SOURCES = []

def get_video_sources(stream_flag=0):
    """Return video sources based on stream flag."""
    if stream_flag:
        return (STANDARD_RTSP_SOURCES + 
                TOWER_RTSP_SOURCES + 
                ENTRANCE_RTSP_SOURCES + 
                EXIT_RTSP_SOURCES)
    return VIDEO_SOURCES

__all__ = [
    'DeviceConfig',
    'DOOR_MODEL_FILENAME',
    'YOLO_HUMAN_MODEL_FILENAME',
    'LIGHT_MODEL_FILENAME',
    'LABCOAT_MODEL_FILENAME',
    'OBJECT_MODEL_FILENAME', # Added for the new object exit model
    'STREAM',
    'STANDARD_VIDEO_FILES',
    'TOWER_VIDEO_FILES',
    'ENTRANCE_VIDEO_FILES',
    'EXIT_VIDEO_FILES',
    'STANDARD_RTSP_SOURCES',
    'TOWER_RTSP_SOURCES',
    'ENTRANCE_RTSP_SOURCES',
    'EXIT_RTSP_SOURCES',
    'HELPERS_STATIC_CONFIG',
    'VIDEO_SOURCES',
    'get_video_sources'
]