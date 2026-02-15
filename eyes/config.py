"""Configuration module for backward compatibility."""
import time  # Added time import
from .config import DeviceConfig

# Import configuration from the config package
from .config import (
    DOOR_MODEL_FILENAME,
    YOLO_HUMAN_MODEL_FILENAME,
    LIGHT_MODEL_FILENAME,
    LABCOAT_MODEL_FILENAME,
    OBJECT_MODEL_FILENAME,  # Added for the new object exit model
    STREAM,
    STANDARD_VIDEO_FILES,
    TOWER_VIDEO_FILES,
    ENTRANCE_VIDEO_FILES,
    EXIT_VIDEO_FILES,
    STANDARD_RTSP_SOURCES,
    TOWER_RTSP_SOURCES,
    ENTRANCE_RTSP_SOURCES,
    EXIT_RTSP_SOURCES,
    HELPERS_STATIC_CONFIG,
    VIDEO_SOURCES,
    get_video_sources,
)

# Default value for headless mode, can be overridden by other config mechanisms if implemented
HEADLESS_MODE = False

__all__ = [
    'DeviceConfig',
    'DOOR_MODEL_FILENAME',
    'YOLO_HUMAN_MODEL_FILENAME',
    'LIGHT_MODEL_FILENAME',
    'LABCOAT_MODEL_FILENAME',
    'OBJECT_MODEL_FILENAME',  # Added for the new object exit model
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
    'get_video_sources',
    'logger', # Added logger
    'server_address', # Added server_address
    'sio_client', # Added sio_client
    'device_id', # Added device_id
    'global_toggles', # Added global_toggles
    'time' # Added time
]

# -------------------------------
# SEND CAMERA METRICS FUNCTION (Fixed Version)
# -------------------------------
def send_camera_metrics(channel_idx, people_count, lab_coat_count, green_intervals=None, extra_metrics=None):
    global sio_client, device_id, global_toggles, logger, server_address  # Added logger and server_address to globals
    
    if not sio_client or not device_id or not sio_client.connected:
        logger.debug(f"Cannot send metrics - socket connection: {sio_client and sio_client.connected}, device_id: {device_id}")
        return

    try:
        # Apply toggles to metrics
        toggles = global_toggles.get(channel_idx, {})
        
        # Format metrics according to what the server expects
        metrics = {
            "people_count": people_count if toggles.get('human', False) else 0,
            "lab_coat_count": lab_coat_count if toggles.get('labcoat', False) else 0
        }
        
        # Add tower light metrics if available
        if green_intervals:
            simplified_intervals = {}
            for interval in green_intervals:
                if isinstance(interval, dict) and 'id' in interval:
                    simplified_intervals[str(interval['id'])] = {
                        'id': interval['id'],
                        'duration': round(interval['duration'], 1) if 'duration' in interval else 0
                    }
            if simplified_intervals:
                metrics["green_intervals"] = simplified_intervals

        # Add additional metrics if available
        if extra_metrics:
            # For entrance monitoring
            if toggles.get('entrance', False):
                if 'entrance_in' in extra_metrics:
                    metrics["entrance_in"] = extra_metrics["entrance_in"]
                if 'entrance_out' in extra_metrics:
                    metrics["entrance_out"] = extra_metrics["entrance_out"]
                if 'illegal_entry' in extra_metrics:
                    metrics["illegal_entry"] = extra_metrics["illegal_entry"]
                if 'labcoat_entries' in extra_metrics:
                    metrics["labcoat_entries"] = extra_metrics["labcoat_entries"]
            
            # For exit monitoring
            if toggles.get('exit', False):
                if 'door_status' in extra_metrics:
                    metrics["door_status"] = extra_metrics["door_status"]
                if 'illegal_exit' in extra_metrics:
                    metrics["illegal_exit"] = extra_metrics["illegal_exit"]
                    
            # For tower light
            if toggles.get('light', False):
                if 'tower_light' in extra_metrics:
                    metrics["tower_light"] = extra_metrics["tower_light"]
                if 'green_time' in extra_metrics:
                    metrics["green_time"] = extra_metrics["green_time"]
                if 'green_active' in extra_metrics:
                    metrics["green_active"] = extra_metrics["green_active"]
        
        # Send all metrics to the server
        data = {
            "device_id": device_id,
            "channel": channel_idx,
            "metrics": metrics,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        }
        
        logger.info(f"Sending metrics to server for channel {channel_idx}: {metrics}")
        sio_client.emit("update_metrics", data)
        
    except Exception as e:
        logger.error(f"Error sending camera metrics: {e}")
        
        # Try to reconnect if connection is lost
        if sio_client and not sio_client.connected:
            logger.info("Socket disconnected, attempting to reconnect...")
            try:
                sio_client.connect(f"http://{server_address}:5000")
            except Exception as reconnect_error:
                logger.error(f"Failed to reconnect: {reconnect_error}")
