# anomaly_manager.py

import time
from PyQt5.QtCore import QObject, pyqtSignal, QMutex, QMutexLocker

class AnomalyManager(QObject):
    """
    A tiny thread-safe FIFO buffer (max_items = 4) that broadcasts every
    new anomaly message through the Qt signal `anomaly_added`.
    """
    anomaly_added = pyqtSignal(str)          # emitted with the *full* entry text

    def __init__(self, max_items: int = 4):
        super().__init__()
        self._lock      = QMutex()
        self._entries   = []                 # oldest → newest
        self._max_items = max_items

    # ------------------------------------------------------------------
    def add(self, msg: str):
        """
        Push a new anomaly message; keep only the latest `max_items`.
        """
        ts   = time.strftime("%H:%M:%S")     # real-world wall-clock time
        line = f"{msg} – {ts}"

        with QMutexLocker(self._lock):
            self._entries.append(line)
            if len(self._entries) > self._max_items:
                self._entries.pop(0)

        self.anomaly_added.emit(line)

    # ------------------------------------------------------------------
    def entries(self):
        with QMutexLocker(self._lock):
            return list(self._entries)

