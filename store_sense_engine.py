"""
StoreSense Phase 2: Core Vision Loop & State Machine (V3 Pick/Reject)
=====================================================================

This module implements the StoreSenseEngine class which:
1. Loads configuration from Phase 1's config.json (including tripwire data)
2. Connects to RTSP/HTTP video stream with retry logic
3. Runs the main vision processing loop with MediaPipe Hands + MOG2
4. Detects TRIPWIRE CROSSING with shelf side awareness (left/right)
5. Tracks PICKED vs REJECTED items using decision window logic
6. Tracks Discovery Friction (Neglect Rate per zone)
7. Outputs JSON telemetry periodically

V3 Pick/Reject Logic:
- Vertical tripwire line with configurable shelf side (left or right)
- Hand crosses INTO shelf → tracking starts
- Hand crosses OUT (back to customer side) → 5s decision window starts
- If hand returns and exits again within 5s → REJECTED (item put back)
- If 5s expires without return → PICKED (item taken)

Architecture Note:
- NO object detection models (YOLO, etc.) are used
- Uses MediaPipe Hands for hand tracking
- Uses OpenCV MOG2 Background Subtractor for pixel change detection
- Asynchronous timer tracking ensures video loop never blocks

Key Concepts:
- Tripwire: Vertical line defining shelf edge
- Shelf Side: Which side of tripwire is the shelf (left or right)
- Decision Window: 5s timer after hand exits to determine PICKED/REJECTED
- Discovery Friction (Neglect): How long zones go untouched

Author: StoreSense Team
Version: 3.0 (Pick/Reject)
"""

import cv2
import json
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from enum import Enum, auto
from collections import deque
import threading

# Optional: Import telemetry sender for Phase 4 integration
try:
    from telemetry_sender import TelemetrySender, create_telemetry_sender
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False

# Optional: Import telemetry queue for Phase 5 offline resilience
try:
    from telemetry_queue import TelemetryQueue, create_telemetry_queue
    TELEMETRY_QUEUE_AVAILABLE = True
except ImportError:
    TELEMETRY_QUEUE_AVAILABLE = False

# Import requests for direct API calls (Phase 4 glue)
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Import YOLOv8 for object detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class InteractionEvent(Enum):
    """Types of interaction events detected in a zone."""
    PICKED = auto()     # Item was picked (hand exited, no return within decision window)
    REJECTED = auto()   # Item was rejected (hand returned and exited within decision window)
    TOUCH = auto()      # Hand touched zone but no change detected
    NONE = auto()       # No event
    # Legacy aliases for backwards compatibility
    TAKEN = auto()      # Alias for PICKED
    PUT_BACK = auto()   # Alias for REJECTED


class ZoneState(Enum):
    """State machine states for each ROI zone (V3 Pick/Reject)."""
    IDLE = auto()              # No hand interaction
    HAND_IN_ZONE = auto()      # Hand is currently in the ROI zone
    DECISION_WINDOW = auto()   # Hand exited zone, waiting to see if it returns (5s)
    # Legacy states (kept for potential backwards compatibility)
    HAND_IN_SHELF = auto()     # Legacy: Hand in shelf side (now using HAND_IN_ZONE)
    HAND_PRESENT = auto()      # Alias for HAND_IN_ZONE  
    WAITING_TIMER = auto()     # Legacy: friction window
    STILLNESS_CHECK = auto()   # Legacy: checking for stillness
    ANALYZING = auto()         # Legacy: analyzing pixel changes


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DetectedObject:
    """Represents an object detected by YOLO within a zone."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int]  # Center point of bbox
    
    @property
    def area(self) -> int:
        """Calculate bounding box area."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

@dataclass
class GlobalSettings:
    """Global store settings from config.json."""
    store_open_time: str
    store_close_time: str
    interaction_friction_window: int  # Also referred to as put_back_timer_sec
    decision_window: int = 5  # Seconds to wait before confirming PICKED
    
    @classmethod
    def from_dict(cls, data: dict) -> 'GlobalSettings':
        return cls(
            store_open_time=data.get("store_open_time", "08:00"),
            store_close_time=data.get("store_close_time", "22:00"),
            interaction_friction_window=data.get("interaction_friction_window", 
                                                  data.get("put_back_timer_sec", 10)),
            decision_window=data.get("decision_window", 5)
        )


@dataclass
class ROIConfig:
    """ROI configuration from config.json with tripwire and shelf side."""
    zone_id: str
    x: int
    y: int
    width: int
    height: int
    tripwire: Optional[List[Tuple[int, int]]] = None  # [(x1,y1), (x2,y2)] vertical line
    shelf_side: str = "right"  # Which side of tripwire is the shelf: "left" or "right"
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ROIConfig':
        tripwire = data.get("tripwire")
        if tripwire:
            tripwire = [(p[0], p[1]) for p in tripwire]
        return cls(
            zone_id=data["zone_id"],
            x=data["x"],
            y=data["y"],
            width=data["width"],
            height=data["height"],
            tripwire=tripwire,
            shelf_side=data.get("shelf_side", "right")
        )
    
    def get_bounds(self) -> Tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) bounds."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    def get_area(self) -> int:
        """Return total pixel area of ROI."""
        return self.width * self.height
    
    def has_tripwire(self) -> bool:
        """Check if this ROI has a tripwire defined."""
        return self.tripwire is not None and len(self.tripwire) == 2
    
    def get_shelf_normal(self) -> Optional[Tuple[float, float]]:
        """
        Get the normal vector pointing TOWARD the shelf side.
        
        For vertical tripwire lines:
        - If shelf_side="right": normal points right (+1, 0)
        - If shelf_side="left": normal points left (-1, 0)
        
        Returns:
            Tuple (nx, ny) normalized vector pointing toward shelf, or None if no tripwire.
        """
        if not self.has_tripwire():
            return None
        
        # For vertical lines, use simple left/right logic based on shelf_side
        if self.shelf_side == "right":
            return (1.0, 0.0)  # Points right toward shelf
        else:
            return (-1.0, 0.0)  # Points left toward shelf
    
    def get_tripwire_normal(self) -> Optional[Tuple[float, float]]:
        """
        Get the outward-pointing normal vector of the tripwire.
        Points from shelf toward customer (away from shelf).
        
        This is the OPPOSITE of get_shelf_normal().
        """
        shelf_normal = self.get_shelf_normal()
        if shelf_normal is None:
            return None
        # Outward = opposite of shelf direction
        return (-shelf_normal[0], -shelf_normal[1])


@dataclass
class ZoneTracker:
    """
    Tracks the state and metrics for a single ROI zone (V5 with Improved Object Detection).
    
    V5 State Machine:
    - IDLE: Waiting for hand to enter zone
    - HAND_IN_ZONE: Hand is inside the ROI zone, tracking interaction
    - DECISION_WINDOW: Hand exited zone boundary, 5s window to determine PICKED vs REJECTED
    
    Key tracking:
    - Hand entry/exit from ROI boundary
    - Background subtraction based object change detection (works with ANY objects)
    - Hand trail for visualization
    - Decision window timing
    - Pick/Reject counts
    """
    zone_id: str
    roi: ROIConfig
    state: ZoneState = ZoneState.IDLE
    
    # Timing
    last_touched_timestamp: float = field(default_factory=time.time)
    hand_enter_timestamp: float = 0.0       # When hand entered zone
    hand_exit_timestamp: float = 0.0        # When hand exited zone
    decision_window_end: float = 0.0        # When 5s decision window expires
    
    # V4: Tracking hand returns during decision window
    hand_returned_during_window: bool = False  # Did hand go back into zone?
    hand_exited_after_return: bool = False     # Did hand exit again after returning?
    
    # Hand position tracking (for trail visualization)
    last_hand_position: Optional[Tuple[float, float]] = None
    hand_in_shelf_side: bool = False  # Legacy: kept for compatibility
    hand_in_zone: bool = False  # Is hand currently inside ROI boundary?
    hand_trail: deque = field(default_factory=lambda: deque(maxlen=30))  # Trail of hand positions
    
    # Environment comparison tracking
    baseline_frame: Optional[np.ndarray] = None  # Continuously refreshed idle-scene baseline
    interaction_baseline_frame: Optional[np.ndarray] = None  # Scene before current hand interaction
    baseline_contours: List[Any] = field(default_factory=list)
    object_change_detected: bool = False  # Scene differs from pre-interaction baseline
    change_magnitude: float = 0.0  # How much the environment changed (0-100%)
    
    # Object detection tracking (YOLO based - optional)
    baseline_objects: List[DetectedObject] = field(default_factory=list)  # Objects before hand entry
    current_objects: List[DetectedObject] = field(default_factory=list)   # Current detected objects
    picked_object: Optional[DetectedObject] = None  # Object being picked (for trail visualization)
    last_object_detection_time: float = 0.0  # Timestamp of last object detection
    
    # Event tracking
    recent_events: deque = field(default_factory=lambda: deque(maxlen=10))
    total_picked: int = 0       # Items taken (not returned within window)
    total_rejected: int = 0     # Items put back (returned within window)
    total_touches: int = 0      # Interactions that were just touches
    total_tripwire_crosses: int = 0
    
    # Legacy aliases for compatibility
    total_taken: int = 0        # Alias for total_picked
    total_put_back: int = 0     # Alias for total_rejected
    
    # Flags
    mog2_paused: bool = False
    
    def reset_interaction(self) -> None:
        """Reset interaction tracking for new cycle."""
        self.state = ZoneState.IDLE
        self.hand_enter_timestamp = 0.0
        self.hand_exit_timestamp = 0.0
        self.decision_window_end = 0.0
        self.hand_returned_during_window = False
        self.hand_exited_after_return = False
        self.mog2_paused = False
        self.hand_in_shelf_side = False
        self.hand_in_zone = False
        self.hand_trail.clear()
        self.picked_object = None
        self.baseline_objects = []
        self.current_objects = []
        # V5: Reset background subtraction fields
        self.baseline_frame = None
        self.interaction_baseline_frame = None
        self.baseline_contours = []
        self.object_change_detected = False
        self.change_magnitude = 0.0
    
    def add_hand_position(self, position: Tuple[float, float]) -> None:
        """Add a hand position to the trail."""
        self.hand_trail.append(position)
        self.last_hand_position = position
    
    def add_event(self, event: InteractionEvent) -> None:
        """Record an interaction event."""
        event_record = {
            "event": event.name,
            "timestamp": datetime.now().isoformat(),
            "zone_id": self.zone_id
        }
        self.recent_events.append(event_record)
        
        if event == InteractionEvent.PICKED or event == InteractionEvent.TAKEN:
            self.total_picked += 1
            self.total_taken += 1  # Legacy alias
        elif event == InteractionEvent.REJECTED or event == InteractionEvent.PUT_BACK:
            self.total_rejected += 1
            self.total_put_back += 1  # Legacy alias
        elif event == InteractionEvent.TOUCH:
            self.total_touches += 1
    
    def get_idle_time(self) -> float:
        """Get seconds since last touch."""
        return time.time() - self.last_touched_timestamp
    
    def get_neglect_rate(self, store_open_seconds: float) -> float:
        """
        Calculate neglect rate as percentage of store open time.
        
        Args:
            store_open_seconds: Total seconds store has been open today
            
        Returns:
            Percentage (0-100) of time this zone has been idle
        """
        if store_open_seconds <= 0:
            return 0.0
        idle_time = self.get_idle_time()
        return min(100.0, (idle_time / store_open_seconds) * 100)


@dataclass 
class TelemetryPayload:
    """JSON telemetry payload structure."""
    timestamp: str
    store_status: str
    zones: List[Dict[str, Any]]
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "store_status": self.store_status,
            "zones": self.zones
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# =============================================================================
# MAIN ENGINE CLASS
# =============================================================================

class StoreSenseEngine:
    """
    Core vision processing engine for StoreSense.
    
    This class orchestrates:
    1. Config loading and validation
    2. Video stream connection with retry logic
    3. MediaPipe Hands initialization and processing
    4. MOG2 Background Subtractor with per-ROI pause mechanism
    5. State machine for each zone tracking interactions
    6. Conversion Friction (TAKEN/PUT_BACK) detection
    7. Discovery Friction (Neglect Rate) calculation
    8. Periodic JSON telemetry output
    
    Architecture:
    - Main loop runs synchronously but uses async time tracking
    - MOG2 learning rate is dynamically set per-ROI based on hand presence
    - Stillness gate prevents false positives from lingering motion
    """
    
    # Display colors (BGR)
    COLOR_IDLE = (128, 128, 128)       # Gray - no activity
    COLOR_HAND_PRESENT = (0, 255, 255)  # Yellow - hand in zone
    COLOR_ANALYZING = (255, 0, 255)     # Magenta - analyzing
    COLOR_TAKEN = (0, 0, 255)           # Red - item taken
    COLOR_PUT_BACK = (0, 255, 0)        # Green - item put back
    COLOR_HAND_BOX = (255, 165, 0)      # Orange - hand bounding box
    
    # Thresholds
    STILLNESS_THRESHOLD = 0.05   # 5% of ROI area for stillness check
    AREA_CHANGE_THRESHOLD = 0.15  # 15% change to detect TAKEN vs PUT_BACK
    MIN_HAND_AREA = 150          # Minimum pixels to consider valid hand
    
    def __init__(
        self,
        config_path: str = "config.json",
        telemetry_interval: float = 5.0,
        show_display: bool = True,
        max_retries: int = 5,
        retry_delay: float = 2.0,
        api_url: str = "http://localhost:3001",
        enable_api_telemetry: bool = True,
        use_offline_queue: bool = True,
        queue_db_path: Optional[str] = None,
        yolo_model_path: str = "yolov8n.pt"
    ):
        """
        Initialize the StoreSense Engine.
        
        Args:
            config_path: Path to config.json from Phase 1
            telemetry_interval: Seconds between telemetry outputs
            show_display: Whether to show OpenCV visualization window
            max_retries: Max connection retry attempts
            retry_delay: Seconds between retries
            api_url: URL for Phase 4/5 telemetry API
            enable_api_telemetry: Whether to send telemetry to API
            use_offline_queue: Use Phase 5 SQLite queue for offline resilience
            queue_db_path: Path for offline queue database (default: telemetry_queue.db)
        """
        self.config_path = Path(config_path)
        self.telemetry_interval = telemetry_interval
        self.show_display = show_display
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.api_url = api_url
        self.enable_api_telemetry = enable_api_telemetry
        self.use_offline_queue = use_offline_queue
        self.queue_db_path = queue_db_path
        self.yolo_model_path = yolo_model_path
        
        # Configuration (loaded from file)
        self.global_settings: Optional[GlobalSettings] = None
        self.rtsp_url: Optional[str] = None
        
        # Zone trackers (one per ROI)
        self.zone_trackers: Dict[str, ZoneTracker] = {}
        
        # Video capture
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_width: int = 0
        self.frame_height: int = 0
        
        # MediaPipe Hands
        self.mp_hands = None
        self.hand_tracker = None
        
        # MOG2 Background Subtractor
        self.bg_subtractor = None
        
        # Timing
        self.last_telemetry_time: float = 0.0
        self.store_open_timestamp: float = 0.0
        self.engine_start_time: float = 0.0
        
        # Control flags
        self.is_running: bool = False
        self.is_paused: bool = False  # True when outside store hours
        
        # Phase 5: Offline-capable Telemetry Queue (preferred)
        self.telemetry_queue: Optional[TelemetryQueue] = None
        if enable_api_telemetry and use_offline_queue and TELEMETRY_QUEUE_AVAILABLE:
            self.telemetry_queue = create_telemetry_queue(
                db_path=queue_db_path,
                api_url=api_url
            )
            logger.info(f"Telemetry queue initialized (offline-capable): {api_url}")
        
        # Phase 4: Direct Telemetry sender (fallback if queue not available)
        self.telemetry_sender: Optional[TelemetrySender] = None
        if enable_api_telemetry and not self.telemetry_queue and TELEMETRY_AVAILABLE:
            self.telemetry_sender = create_telemetry_sender(api_url=api_url)
            logger.info(f"Telemetry sender initialized: {api_url}")
        elif enable_api_telemetry and not self.telemetry_queue and not TELEMETRY_AVAILABLE:
            logger.warning("No telemetry module available - API telemetry disabled")
        
        logger.info("StoreSenseEngine initialized")
    
    # =========================================================================
    # CONFIGURATION LOADING
    # =========================================================================
    
    def load_config(self) -> bool:
        """
        Load configuration from Phase 1's config.json.
        
        Returns:
            bool: True if config loaded successfully
        """
        if not self.config_path.exists():
            logger.error(f"Config file not found: {self.config_path}")
            return False
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Load global settings
            self.global_settings = GlobalSettings.from_dict(
                config.get("global_settings", {})
            )
            
            # Load RTSP URL
            self.rtsp_url = config.get("rtsp_url", 0)
            
            # Load ROIs and create zone trackers
            rois_data = config.get("rois", [])
            for roi_data in rois_data:
                roi = ROIConfig.from_dict(roi_data)
                self.zone_trackers[roi.zone_id] = ZoneTracker(
                    zone_id=roi.zone_id,
                    roi=roi
                )
            
            logger.info(f"Config loaded successfully:")
            logger.info(f"  - Store hours: {self.global_settings.store_open_time} - "
                       f"{self.global_settings.store_close_time}")
            logger.info(f"  - Put-back timer: {self.global_settings.interaction_friction_window}s")
            logger.info(f"  - ROI zones: {len(self.zone_trackers)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return False
    
    # =========================================================================
    # TIME-AWARE ANALYTICS
    # =========================================================================
    
    def is_store_open(self) -> bool:
        """
        Check if current time is within store operating hours.
        
        Returns:
            bool: True if store is currently open
        """
        if self.global_settings is None:
            return True  # Default to open if no settings
        
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        
        open_time = self.global_settings.store_open_time
        close_time = self.global_settings.store_close_time
        
        return open_time <= current_time <= close_time
    
    def get_store_open_seconds(self) -> float:
        """
        Calculate how many seconds the store has been open today.
        
        Returns:
            float: Seconds since store opened (or since engine started if after open)
        """
        if self.global_settings is None:
            return time.time() - self.engine_start_time
        
        now = datetime.now()
        
        # Parse store open time
        open_hour, open_min = map(int, self.global_settings.store_open_time.split(':'))
        store_open_today = now.replace(hour=open_hour, minute=open_min, second=0, microsecond=0)
        
        # If we started after store opened, use engine start time
        if self.engine_start_time > store_open_today.timestamp():
            return time.time() - self.engine_start_time
        
        return time.time() - store_open_today.timestamp()

    def fit_rois_to_frame(self) -> None:
        """Scale ROIs down when calibration coordinates exceed current frame size."""
        if not self.zone_trackers or self.frame_width <= 0 or self.frame_height <= 0:
            return

        max_x2 = max(tracker.roi.x + tracker.roi.width for tracker in self.zone_trackers.values())
        max_y2 = max(tracker.roi.y + tracker.roi.height for tracker in self.zone_trackers.values())

        if max_x2 <= self.frame_width and max_y2 <= self.frame_height:
            return

        scale = min(self.frame_width / max_x2, self.frame_height / max_y2)
        logger.warning(
            f"ROI coordinates exceed frame {self.frame_width}x{self.frame_height}; auto-scaling zones by {scale:.3f}"
        )

        for tracker in self.zone_trackers.values():
            roi = tracker.roi
            roi.x = max(0, int(round(roi.x * scale)))
            roi.y = max(0, int(round(roi.y * scale)))
            roi.width = max(40, int(round(roi.width * scale)))
            roi.height = max(40, int(round(roi.height * scale)))

            if roi.tripwire:
                roi.tripwire = [
                    (int(round(px * scale)), int(round(py * scale)))
                    for px, py in roi.tripwire
                ]

            if roi.x + roi.width > self.frame_width:
                roi.width = max(40, self.frame_width - roi.x)
            if roi.y + roi.height > self.frame_height:
                roi.height = max(40, self.frame_height - roi.y)
    
    # =========================================================================
    # VIDEO STREAM CONNECTION
    # =========================================================================
    
    def connect(self) -> bool:
        """
        Connect to video stream with retry logic.
        
        Supports:
        - Local webcam (index 0, 1, etc.)
        - RTSP streams
        - HTTP streams (IP Webcam app)
        
        Returns:
            bool: True if connected successfully
        """
        for attempt in range(1, self.max_retries + 1):
            logger.info(f"Connection attempt {attempt}/{self.max_retries}...")
            
            try:
                # Determine capture method based on URL type
                if isinstance(self.rtsp_url, int) or (isinstance(self.rtsp_url, str) and self.rtsp_url.isdigit()):
                    # Local webcam
                    url = int(self.rtsp_url) if isinstance(self.rtsp_url, str) else self.rtsp_url
                    self.cap = cv2.VideoCapture(url, cv2.CAP_DSHOW)
                    logger.info(f"Using local webcam: {url}")
                elif self.rtsp_url.startswith('http://') or self.rtsp_url.startswith('https://'):
                    # HTTP stream (IP Webcam app)
                    self.cap = cv2.VideoCapture(self.rtsp_url)
                    logger.info(f"Using HTTP stream: {self.rtsp_url}")
                else:
                    # RTSP stream
                    self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                    logger.info(f"Using RTSP stream: {self.rtsp_url}")
                
                # Set buffer size to minimize latency
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self.frame_height, self.frame_width = frame.shape[:2]
                        self.fit_rois_to_frame()
                        logger.info(f"Connected! Frame size: {self.frame_width}x{self.frame_height}")
                        return True
                        
            except Exception as e:
                logger.error(f"Connection error: {e}")
            
            if self.cap:
                self.cap.release()
                self.cap = None
            
            if attempt < self.max_retries:
                time.sleep(self.retry_delay)
        
        logger.error("Failed to connect to video stream")
        return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame with error handling."""
        if self.cap is None or not self.cap.isOpened():
            if not self.connect():
                return False, None
        
        ret, frame = self.cap.read()
        if ret and frame is not None:
            return True, frame
        
        # Try reconnecting on failure
        logger.warning("Frame read failed, reconnecting...")
        self.cap.release()
        if self.connect():
            return self.cap.read()
        
        return False, None
    
    # =========================================================================
    # MODEL INITIALIZATION
    # =========================================================================
    
    def initialize_models(self) -> bool:
        """
        Initialize MediaPipe Hands and MOG2 Background Subtractor.
        
        Supports both legacy MediaPipe API (solutions) and new Tasks API (0.10+).
        
        Returns:
            bool: True if all models initialized
        """
        success = True
        
        # Initialize MediaPipe Hands
        try:
            import mediapipe as mp
            
            # Try new Tasks API first (MediaPipe 0.10+)
            try:
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                import urllib.request
                import os
                
                # Download hand landmarker model if not present
                model_path = "hand_landmarker.task"
                if not os.path.exists(model_path):
                    logger.info("Downloading MediaPipe hand landmarker model...")
                    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                    urllib.request.urlretrieve(url, model_path)
                    logger.info("Model downloaded successfully")
                
                # Create hand landmarker with Tasks API
                base_options = python.BaseOptions(model_asset_path=model_path)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=2,
                    min_hand_detection_confidence=0.45,
                    min_tracking_confidence=0.35
                )
                self.hand_tracker = vision.HandLandmarker.create_from_options(options)
                self.mp_hands = None  # Not using legacy API
                self._use_tasks_api = True
                logger.info("MediaPipe Hands initialized (Tasks API)")
                
            except Exception as tasks_error:
                # Fall back to legacy solutions API
                logger.info(f"Tasks API not available ({tasks_error}), trying legacy API...")
                self.mp_hands = mp.solutions.hands
                self.hand_tracker = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.45,
                    min_tracking_confidence=0.35
                )
                self._use_tasks_api = False
                logger.info("MediaPipe Hands initialized (Legacy API)")
                
        except Exception as e:
            logger.error(f"MediaPipe initialization failed: {e}")
            success = False
        
        # Initialize MOG2 Background Subtractor
        try:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500,
                varThreshold=50,
                detectShadows=True
            )
            logger.info("MOG2 Background Subtractor initialized")
        except Exception as e:
            logger.error(f"MOG2 initialization failed: {e}")
            success = False
        
        # Initialize YOLOv8 for object detection
        try:
            if YOLO_AVAILABLE:
                # Use custom YOLO weights when provided, otherwise fall back to nano model
                self.yolo_model = YOLO(self.yolo_model_path)
                self.yolo_model.to('cpu')  # Use CPU by default, change to 'cuda' if GPU available
                logger.info(f"YOLOv8 object detection initialized: {self.yolo_model_path}")
            else:
                self.yolo_model = None
                logger.warning("YOLOv8 not available - object detection disabled")
        except Exception as e:
            logger.error(f"YOLOv8 initialization failed: {e}")
            self.yolo_model = None
        
        # Initialize frame timestamp counter for Tasks API
        self._frame_timestamp_ms = 0
        
        return success
    
    # =========================================================================
    # HAND DETECTION
    # =========================================================================
    
    def detect_hands(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect hands in frame using MediaPipe.
        
        Supports both legacy API (solutions.hands) and new Tasks API (HandLandmarker).
        
        Args:
            frame: BGR frame from video
            
        Returns:
            List of hand bounding boxes as (x, y, w, h)
        """
        if self.hand_tracker is None:
            return []
        
        hand_boxes = []
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            if hasattr(self, '_use_tasks_api') and self._use_tasks_api:
                # New Tasks API (MediaPipe 0.10+)
                from mediapipe.tasks.python import vision
                import mediapipe as mp
                
                # Create MediaPipe Image
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # Detect hands with timestamp (required for VIDEO mode)
                self._frame_timestamp_ms += 33  # ~30fps
                results = self.hand_tracker.detect_for_video(mp_image, self._frame_timestamp_ms)
                
                # Extract bounding boxes from landmarks
                if results.hand_landmarks:
                    for hand_landmarks in results.hand_landmarks:
                        x_coords = [lm.x for lm in hand_landmarks]
                        y_coords = [lm.y for lm in hand_landmarks]
                        
                        x_min = int(min(x_coords) * self.frame_width)
                        x_max = int(max(x_coords) * self.frame_width)
                        y_min = int(min(y_coords) * self.frame_height)
                        y_max = int(max(y_coords) * self.frame_height)
                        
                        # Add padding so partial hands still form a usable box
                        padding = 35
                        x_min = max(0, x_min - padding)
                        y_min = max(0, y_min - padding)
                        x_max = min(self.frame_width, x_max + padding)
                        y_max = min(self.frame_height, y_max + padding)
                        
                        width = x_max - x_min
                        height = y_max - y_min
                        
                        if width * height >= self.MIN_HAND_AREA:
                            hand_boxes.append((x_min, y_min, width, height))
            else:
                # Legacy solutions API
                results = self.hand_tracker.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        x_coords = [lm.x for lm in hand_landmarks.landmark]
                        y_coords = [lm.y for lm in hand_landmarks.landmark]
                        
                        x_min = int(min(x_coords) * self.frame_width)
                        x_max = int(max(x_coords) * self.frame_width)
                        y_min = int(min(y_coords) * self.frame_height)
                        y_max = int(max(y_coords) * self.frame_height)
                        
                        # Add padding so partial hands still form a usable box
                        padding = 35
                        x_min = max(0, x_min - padding)
                        y_min = max(0, y_min - padding)
                        x_max = min(self.frame_width, x_max + padding)
                        y_max = min(self.frame_height, y_max + padding)
                        
                        width = x_max - x_min
                        height = y_max - y_min
                        
                        if width * height >= self.MIN_HAND_AREA:
                            hand_boxes.append((x_min, y_min, width, height))
                            
        except Exception as e:
            logger.debug(f"Hand detection error: {e}")
        
        return hand_boxes
    
    # =========================================================================
    # OBJECT DETECTION (YOLOv8)
    # =========================================================================
    
    def detect_objects_in_roi(
        self,
        frame: np.ndarray,
        roi: ROIConfig,
        confidence_threshold: float = 0.25
    ) -> List[DetectedObject]:
        """
        Detect objects within an ROI using YOLOv8.
        
        Args:
            frame: Full video frame
            roi: ROI configuration defining the area to scan
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            List of DetectedObject instances found in the ROI
        """
        if self.yolo_model is None:
            return []
        
        try:
            # Crop the ROI from frame
            x1, y1, x2, y2 = roi.get_bounds()
            roi_crop = frame[y1:y2, x1:x2]
            
            if roi_crop.size == 0:
                return []
            
            # Run YOLO inference
            results = self.yolo_model(roi_crop, verbose=False, conf=confidence_threshold)
            
            detected_objects = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                    
                for i, box in enumerate(boxes):
                    # Get box coordinates (relative to ROI crop)
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = result.names[cls_id]
                    
                    # Convert to absolute frame coordinates
                    abs_x1 = int(xyxy[0]) + x1
                    abs_y1 = int(xyxy[1]) + y1
                    abs_x2 = int(xyxy[2]) + x1
                    abs_y2 = int(xyxy[3]) + y1
                    
                    # Calculate center
                    center_x = (abs_x1 + abs_x2) // 2
                    center_y = (abs_y1 + abs_y2) // 2
                    
                    detected_objects.append(DetectedObject(
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                        bbox=(abs_x1, abs_y1, abs_x2, abs_y2),
                        center=(center_x, center_y)
                    ))
            
            return detected_objects
            
        except Exception as e:
            logger.debug(f"Object detection error: {e}")
            return []
    
    def find_missing_object(
        self,
        baseline: List[DetectedObject],
        current: List[DetectedObject],
        iou_threshold: float = 0.3
    ) -> Optional[DetectedObject]:
        """
        Find which object from baseline is missing in current detections.
        
        Uses IoU (Intersection over Union) to match objects between frames.
        
        Args:
            baseline: Objects detected before hand interaction
            current: Objects detected after hand interaction
            iou_threshold: Minimum IoU to consider objects as same
            
        Returns:
            The missing object if found, None otherwise
        """
        if not baseline:
            return None
        
        def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
            """Calculate Intersection over Union between two boxes."""
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
            
            intersection = (x2 - x1) * (y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        
        # For each baseline object, check if it exists in current
        for base_obj in baseline:
            found_match = False
            for curr_obj in current:
                iou = calculate_iou(base_obj.bbox, curr_obj.bbox)
                if iou >= iou_threshold and base_obj.class_id == curr_obj.class_id:
                    found_match = True
                    break
            
            if not found_match:
                # This object is missing - it was likely picked
                return base_obj
        
        return None
    
    # =========================================================================
    # V5: BACKGROUND SUBTRACTION BASED OBJECT CHANGE DETECTION
    # =========================================================================
    
    def capture_baseline_frame(
        self,
        frame: np.ndarray,
        roi: ROIConfig
    ) -> np.ndarray:
        """
        Capture and preprocess a baseline frame for later comparison.
        
        This captures the ROI region before hand entry to detect changes
        when hand exits (works with ANY objects, not just COCO classes).
        
        Args:
            frame: Full video frame
            roi: ROI configuration
            
        Returns:
            Preprocessed grayscale ROI crop
        """
        x1, y1, x2, y2 = roi.get_bounds()
        roi_crop = frame[y1:y2, x1:x2].copy()
        
        # Convert to grayscale and apply Gaussian blur for noise reduction
        gray = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return blurred
    
    def detect_object_change(
        self,
        baseline_frame: np.ndarray,
        current_frame: np.ndarray,
        roi: ROIConfig,
        threshold: int = 20,
        min_contour_area: int = 120
    ) -> Tuple[bool, float, List[Tuple[int, int, int, int]]]:
        """
        Detect if objects have changed between baseline and current frame.
        
        Uses background subtraction and contour analysis to detect ANY objects
        that have been picked or placed, regardless of their type.
        
        Args:
            baseline_frame: Preprocessed grayscale baseline ROI
            current_frame: Full video frame
            roi: ROI configuration
            threshold: Pixel difference threshold for change detection
            min_contour_area: Minimum contour area to consider as object change
            
        Returns:
            Tuple of (change_detected: bool, change_magnitude: float, changed_regions: list)
            - change_detected: True if significant change detected
            - change_magnitude: Percentage of ROI area that changed (0-100)
            - changed_regions: List of bounding boxes for changed areas
        """
        x1, y1, x2, y2 = roi.get_bounds()
        current_crop = current_frame[y1:y2, x1:x2]
        
        if current_crop.size == 0 or baseline_frame.size == 0:
            return False, 0.0, []
        
        # Convert current to grayscale and blur
        current_gray = cv2.cvtColor(current_crop, cv2.COLOR_BGR2GRAY)
        current_blurred = cv2.GaussianBlur(current_gray, (5, 5), 0)
        
        # Ensure same size
        if baseline_frame.shape != current_blurred.shape:
            return False, 0.0, []
        
        # Calculate absolute difference
        diff = cv2.absdiff(baseline_frame, current_blurred)
        
        # Apply threshold to get binary mask
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours of changed regions
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and collect bounding boxes
        changed_regions = []
        total_changed_pixels = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                # Convert to absolute coordinates
                changed_regions.append((x + x1, y + y1, w, h))
                total_changed_pixels += area
        
        # Calculate change magnitude as percentage of ROI area
        roi_area = roi.get_area()
        change_magnitude = (total_changed_pixels / roi_area * 100) if roi_area > 0 else 0.0
        
        # Significant change threshold: at least 0.2% of ROI changed with at least one contour
        change_detected = len(changed_regions) > 0 and change_magnitude >= 0.2
        
        return change_detected, change_magnitude, changed_regions
    
    def detect_objects_via_contours(
        self,
        frame: np.ndarray,
        roi: ROIConfig,
        min_area: int = 80,
        max_area: int = 50000
    ) -> List[DetectedObject]:
        """
        Detect objects in ROI using contour analysis (works with ANY objects).
        
        This is a fallback method when YOLO doesn't recognize specific objects
        like shuttlecocks. It finds all distinct objects based on edge detection.
        
        Args:
            frame: Full video frame
            roi: ROI configuration
            min_area: Minimum contour area to consider
            max_area: Maximum contour area to consider
            
        Returns:
            List of DetectedObject instances
        """
        x1, y1, x2, y2 = roi.get_bounds()
        roi_crop = frame[y1:y2, x1:x2]
        
        if roi_crop.size == 0:
            return []
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use adaptive thresholding to handle varying lighting
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_objects = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Convert to absolute coordinates
                abs_x1 = x + x1
                abs_y1 = y + y1
                abs_x2 = abs_x1 + w
                abs_y2 = abs_y1 + h
                
                center_x = abs_x1 + w // 2
                center_y = abs_y1 + h // 2
                
                detected_objects.append(DetectedObject(
                    class_id=-1,  # Unknown class
                    class_name="object",  # Generic name
                    confidence=0.8,  # Confidence based on contour detection
                    bbox=(abs_x1, abs_y1, abs_x2, abs_y2),
                    center=(center_x, center_y)
                ))
        
        return detected_objects
    
    def hybrid_object_detection(
        self,
        frame: np.ndarray,
        roi: ROIConfig,
        confidence_threshold: float = 0.2
    ) -> List[DetectedObject]:
        """
        Hybrid object detection combining YOLO and contour analysis.
        
        First tries YOLO for recognized objects, then falls back to
        contour analysis for generic object detection.
        
        Args:
            frame: Full video frame
            roi: ROI configuration
            confidence_threshold: YOLO confidence threshold
            
        Returns:
            List of DetectedObject instances
        """
        yolo_objects = self.detect_objects_in_roi(frame, roi, confidence_threshold)
        contour_objects = self.detect_objects_via_contours(frame, roi)

        if not yolo_objects:
            return contour_objects
        if not contour_objects:
            return yolo_objects

        merged = list(yolo_objects)
        for contour_obj in contour_objects:
            duplicate = False
            for yolo_obj in yolo_objects:
                dx = abs(contour_obj.center[0] - yolo_obj.center[0])
                dy = abs(contour_obj.center[1] - yolo_obj.center[1])
                if dx < 25 and dy < 25:
                    duplicate = True
                    break
            if not duplicate:
                merged.append(contour_obj)

        return merged
    
    # =========================================================================
    # HAND IN ROI BOUNDARY CHECK (V4)
    # =========================================================================
    
    def is_hand_in_roi_boundary(
        self,
        hand_box: Tuple[int, int, int, int],
        roi: ROIConfig,
        margin: int = 10
    ) -> bool:
        """
        Check if hand center is inside the ROI boundary.
        
        Args:
            hand_box: (x, y, w, h) of hand
            roi: ROI configuration
            margin: Pixel margin for entry/exit detection
            
        Returns:
            True if hand center is inside ROI boundary
        """
        hand_center = self.get_hand_center(hand_box)
        x1, y1, x2, y2 = roi.get_bounds()
        
        return (x1 - margin <= hand_center[0] <= x2 + margin and
                y1 - margin <= hand_center[1] <= y2 + margin)
    
    # =========================================================================
    # ROI-HAND INTERSECTION
    # =========================================================================
    
    def check_hand_roi_intersection(
        self,
        hand_box: Tuple[int, int, int, int],
        roi: ROIConfig
    ) -> bool:
        """
        Check if a hand bounding box intersects with an ROI.
        
        Args:
            hand_box: (x, y, w, h) of hand
            roi: ROI configuration
            
        Returns:
            bool: True if intersecting
        """
        hx, hy, hw, hh = hand_box
        rx1, ry1, rx2, ry2 = roi.get_bounds()
        
        # Hand bounds
        hx2 = hx + hw
        hy2 = hy + hh
        
        # Check intersection
        return not (hx2 < rx1 or hx > rx2 or hy2 < ry1 or hy > ry2)
    
    # =========================================================================
    # TRIPWIRE CROSSING DETECTION (V2 UPGRADE)
    # =========================================================================
    
    def get_hand_center(self, hand_box: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """
        Get the center point of a hand bounding box.
        
        Args:
            hand_box: (x, y, w, h) of hand
            
        Returns:
            Tuple (cx, cy) center coordinates
        """
        hx, hy, hw, hh = hand_box
        return (hx + hw / 2, hy + hh / 2)
    
    def get_signed_distance_to_tripwire(
        self,
        point: Tuple[float, float],
        tripwire: List[Tuple[int, int]],
        normal: Tuple[float, float]
    ) -> float:
        """
        Calculate signed distance from a point to the tripwire line.
        
        Positive = on the customer/outside side (in direction of normal)
        Negative = on the shelf/inside side (opposite to normal)
        
        Uses the plane equation: d = (P - P0) · N
        
        Args:
            point: (x, y) point to check
            tripwire: [(x1,y1), (x2,y2)] line endpoints
            normal: (nx, ny) outward-pointing normal vector
            
        Returns:
            Signed distance (positive = outside, negative = inside)
        """
        # Use first tripwire point as reference point on the line
        p0 = tripwire[0]
        
        # Vector from reference point to test point
        dx = point[0] - p0[0]
        dy = point[1] - p0[1]
        
        # Dot product with normal gives signed distance
        return dx * normal[0] + dy * normal[1]
    
    def check_tripwire_crossing(
        self,
        tracker: 'ZoneTracker',
        current_position: Tuple[float, float]
    ) -> Optional[str]:
        """
        Check if a hand has crossed the tripwire between frames.
        
        Detects crossing by comparing signed distances to the tripwire line
        between the previous and current hand positions.
        
        Args:
            tracker: Zone tracker with tripwire and position history
            current_position: Current hand center (x, y)
            
        Returns:
            'outward' if crossed from shelf to customer side
            'inward' if crossed from customer to shelf side
            None if no crossing occurred
        """
        roi = tracker.roi
        
        # Check if tripwire is defined
        if not roi.has_tripwire():
            return None
        
        tripwire = roi.tripwire
        normal = roi.get_tripwire_normal()
        
        if not tripwire or not normal:
            return None
        
        # Need previous position to detect crossing
        if tracker.last_hand_position is None:
            return None
        
        prev_position = tracker.last_hand_position
        
        # Calculate signed distances
        prev_dist = self.get_signed_distance_to_tripwire(prev_position, tripwire, normal)
        curr_dist = self.get_signed_distance_to_tripwire(current_position, tripwire, normal)
        
        # Check for sign change (crossing)
        if prev_dist <= 0 and curr_dist > 0:
            # Crossed from inside (shelf) to outside (customer)
            return 'outward'
        elif prev_dist >= 0 and curr_dist < 0:
            # Crossed from outside (customer) to inside (shelf)
            return 'inward'
        
        return None
    
    def check_hand_tripwire_cross_in_zone(
        self,
        tracker: 'ZoneTracker',
        hand_boxes: List[Tuple[int, int, int, int]]
    ) -> Tuple[bool, Optional[str], Optional[Tuple[float, float]]]:
        """
        Check if any hand in the zone has crossed the tripwire.
        
        Args:
            tracker: Zone tracker
            hand_boxes: List of hand bounding boxes
            
        Returns:
            Tuple of (hand_in_zone, crossing_direction, hand_position)
            crossing_direction is 'outward', 'inward', or None
        """
        roi = tracker.roi
        crossing_direction = None
        current_hand_position = None
        hand_in_zone = False
        
        for hand_box in hand_boxes:
            if self.check_hand_roi_intersection(hand_box, roi):
                hand_in_zone = True
                hand_center = self.get_hand_center(hand_box)
                current_hand_position = hand_center
                
                # Check for tripwire crossing
                crossing = self.check_tripwire_crossing(tracker, hand_center)
                if crossing:
                    crossing_direction = crossing
                    logger.info(f"[{tracker.zone_id}] *** TRIPWIRE CROSSED {crossing.upper()}! ***")
                    print(f"\n*** TRIPWIRE CROSSED! Zone: {tracker.zone_id}, Direction: {crossing.upper()} ***\n")
                
                break  # Use first hand found in zone
        
        return hand_in_zone, crossing_direction, current_hand_position
    
    # =========================================================================
    # MOG2 PROCESSING WITH PAUSE MECHANISM
    # =========================================================================
    
    def process_mog2(
        self,
        frame: np.ndarray,
        hand_boxes: List[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """
        Process frame through MOG2 with per-ROI pause mechanism.
        
        When a hand intersects an ROI, we set learning rate to 0 for that
        region so the hand doesn't get learned as background.
        
        Args:
            frame: Current BGR frame
            hand_boxes: List of detected hand bounding boxes
            
        Returns:
            Foreground mask from MOG2
        """
        if self.bg_subtractor is None:
            return np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # Create a mask of regions where we should pause learning
        pause_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
        
        # For each zone tracker, check if hand is present
        for zone_id, tracker in self.zone_trackers.items():
            roi = tracker.roi
            hand_in_zone = False
            
            for hand_box in hand_boxes:
                if self.check_hand_roi_intersection(hand_box, roi):
                    hand_in_zone = True
                    break
            
            if hand_in_zone:
                # Pause MOG2 learning for this ROI
                tracker.mog2_paused = True
                rx1, ry1, rx2, ry2 = roi.get_bounds()
                pause_mask[ry1:ry2, rx1:rx2] = 0
            else:
                tracker.mog2_paused = False
        
        # Apply MOG2 with selective learning
        # First, apply with normal learning rate
        fg_mask = self.bg_subtractor.apply(frame, learningRate=0.01)
        
        # For paused regions, re-apply with learning rate 0
        if np.any(pause_mask == 0):
            fg_mask_no_learn = self.bg_subtractor.apply(frame, learningRate=0)
            # Use the no-learn mask for paused regions
            fg_mask = np.where(pause_mask == 0, fg_mask_no_learn, fg_mask)
        
        # Clean up the mask (remove shadows marked as 127)
        fg_mask = np.where(fg_mask == 255, 255, 0).astype(np.uint8)
        
        return fg_mask
    
    def calculate_roi_motion_area(
        self,
        fg_mask: np.ndarray,
        roi: ROIConfig
    ) -> float:
        """
        Calculate the total white pixel area in ROI from foreground mask.
        
        Args:
            fg_mask: MOG2 foreground mask
            roi: ROI configuration
            
        Returns:
            Total area of white pixels (motion) in the ROI
        """
        rx1, ry1, rx2, ry2 = roi.get_bounds()
        roi_mask = fg_mask[ry1:ry2, rx1:rx2]
        
        # Find contours and sum their areas
        contours, _ = cv2.findContours(
            roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        total_area = sum(cv2.contourArea(c) for c in contours)
        return total_area
    
    def check_stillness(self, fg_mask: np.ndarray, roi: ROIConfig) -> bool:
        """
        Check if motion in ROI is below stillness threshold.
        
        Args:
            fg_mask: MOG2 foreground mask
            roi: ROI configuration
            
        Returns:
            bool: True if ROI is "still" (motion < 5% of area)
        """
        motion_area = self.calculate_roi_motion_area(fg_mask, roi)
        roi_area = roi.get_area()
        motion_ratio = motion_area / roi_area if roi_area > 0 else 0
        
        return motion_ratio < self.STILLNESS_THRESHOLD

    def detect_hands_from_motion(
        self,
        fg_mask: np.ndarray,
        frame_shape: Tuple[int, int, int],
        min_area: int = 250
    ) -> List[Tuple[int, int, int, int]]:
        """
        Fallback hand detector based on foreground motion contours.

        This is used when MediaPipe misses the hand. It is intentionally generic:
        moving blobs with plausible hand-sized boxes are treated as temporary hand boxes.
        """
        height, width = frame_shape[:2]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[Tuple[int, int, int, int]] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w < 15 or h < 15:
                continue
            if w > width * 0.5 or h > height * 0.5:
                continue

            pad = 12
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(width - x, w + pad * 2)
            h = min(height - y, h + pad * 2)
            boxes.append((x, y, w, h))

        return boxes
    
    # =========================================================================
    # STATE MACHINE PROCESSING (V4 ROI BOUNDARY + OBJECT DETECTION)
    # =========================================================================
    
    def is_point_on_shelf_side(
        self,
        point: Tuple[float, float],
        roi: ROIConfig
    ) -> bool:
        """
        Determine if a point is on the shelf side of the tripwire.
        Legacy function kept for compatibility.
        """
        if not roi.has_tripwire() or roi.tripwire is None:
            return False
        
        tripwire_x = (roi.tripwire[0][0] + roi.tripwire[1][0]) / 2
        
        if roi.shelf_side == "right":
            return point[0] > tripwire_x
        else:
            return point[0] < tripwire_x
    
    def process_zone_state_v4(
        self,
        tracker: ZoneTracker,
        hand_in_zone: bool,
        frame: np.ndarray,
        current_time: float,
        hand_position: Optional[Tuple[float, float]] = None
    ) -> Optional[InteractionEvent]:
        """
        Process one zone using hand presence + environment restoration only.

        Flow:
        - While idle, keep refreshing the baseline scene for the ROI.
        - When a hand enters, freeze that scene as the pre-interaction baseline.
        - When the hand exits, compare the ROI with the frozen baseline.
        - If the ROI changed, start the 5 second window.
        - During the window, if the ROI becomes the same as baseline again,
          count PUT_BACK. If the window expires and the ROI is still changed,
          count PICKED. If no change happened, count TOUCH.
        """
        roi = tracker.roi
        decision_window_sec = self.global_settings.decision_window
        
        event = None
        
        # Update hand position tracking and trail
        if hand_position:
            tracker.add_hand_position(hand_position)
            tracker.hand_in_zone = hand_in_zone
        elif not hand_in_zone:
            tracker.last_hand_position = None
            tracker.hand_in_zone = False
        
        # -----------------------------------------------------------------
        # State: IDLE - Waiting for hand to enter ROI boundary
        # -----------------------------------------------------------------
        if tracker.state == ZoneState.IDLE:
            tracker.baseline_frame = self.capture_baseline_frame(frame, roi)
            tracker.picked_object = None
            
            if hand_in_zone:
                tracker.state = ZoneState.HAND_IN_ZONE
                tracker.hand_enter_timestamp = current_time
                tracker.last_touched_timestamp = current_time
                tracker.hand_trail.clear()
                tracker.interaction_baseline_frame = (
                    tracker.baseline_frame.copy() if tracker.baseline_frame is not None else self.capture_baseline_frame(frame, roi)
                )
                logger.info(f"[{tracker.zone_id}] Hand entered ROI")
        
        # -----------------------------------------------------------------
        # State: HAND_IN_ZONE - Hand is inside ROI, tracking interaction
        # -----------------------------------------------------------------
        elif tracker.state == ZoneState.HAND_IN_ZONE:
            tracker.last_touched_timestamp = current_time
            
            # Check if hand exited ROI boundary
            if not hand_in_zone:
                reference_frame = tracker.interaction_baseline_frame
                if reference_frame is None:
                    reference_frame = tracker.baseline_frame
                if reference_frame is not None:
                    change_detected, change_magnitude, changed_regions = self.detect_object_change(
                        reference_frame, frame, roi
                    )
                else:
                    change_detected, change_magnitude, changed_regions = (False, 0.0, [])

                tracker.object_change_detected = change_detected
                tracker.change_magnitude = change_magnitude

                if changed_regions:
                    rx, ry, rw, rh = max(changed_regions, key=lambda r: r[2] * r[3])
                    tracker.picked_object = DetectedObject(
                        class_id=-1,
                        class_name="changed region",
                        confidence=min(0.99, max(0.5, change_magnitude / 10.0)),
                        bbox=(rx, ry, rx + rw, ry + rh),
                        center=(rx + rw // 2, ry + rh // 2)
                    )

                if change_detected:
                    tracker.state = ZoneState.DECISION_WINDOW
                    tracker.hand_exit_timestamp = current_time
                    tracker.decision_window_end = current_time + decision_window_sec
                    tracker.hand_returned_during_window = False
                    tracker.hand_exited_after_return = False
                    logger.info(f"[{tracker.zone_id}] Hand exited ROI - environment changed ({change_magnitude:.1f}%), starting {decision_window_sec}s timer")
                else:
                    event = InteractionEvent.TOUCH
                    tracker.add_event(event)
                    logger.info(f"[{tracker.zone_id}] EVENT: TOUCH - hand left but environment stayed the same")
                    tracker.reset_interaction()
                    tracker.baseline_frame = self.capture_baseline_frame(frame, roi)
        
        # -----------------------------------------------------------------
        # State: DECISION_WINDOW - 5 second window to determine PICKED/REJECTED
        # -----------------------------------------------------------------
        elif tracker.state == ZoneState.DECISION_WINDOW:
            if hand_in_zone and not tracker.hand_returned_during_window:
                tracker.hand_returned_during_window = True
                logger.info(f"[{tracker.zone_id}] Hand RETURNED to ROI during decision window")

            reference_frame = tracker.interaction_baseline_frame
            if reference_frame is None:
                reference_frame = tracker.baseline_frame
            if reference_frame is not None:
                change_detected, change_magnitude, changed_regions = self.detect_object_change(
                    reference_frame, frame, roi
                )
            else:
                change_detected, change_magnitude, changed_regions = (False, 0.0, [])

            tracker.object_change_detected = change_detected
            tracker.change_magnitude = change_magnitude

            if change_detected and changed_regions:
                rx, ry, rw, rh = max(changed_regions, key=lambda r: r[2] * r[3])
                tracker.picked_object = DetectedObject(
                    class_id=-1,
                    class_name="changed region",
                    confidence=min(0.99, max(0.5, change_magnitude / 10.0)),
                    bbox=(rx, ry, rx + rw, ry + rh),
                    center=(rx + rw // 2, ry + rh // 2)
                )
            elif not change_detected:
                tracker.picked_object = None

            if not change_detected:
                event = InteractionEvent.REJECTED
                tracker.add_event(event)
                logger.info(f"[{tracker.zone_id}] EVENT: REJECTED - environment restored within {decision_window_sec}s")
                self._send_event_to_api(
                    tracker.zone_id,
                    "REJECTED",
                    tracker.get_neglect_rate(self.get_store_open_seconds()),
                    None
                )
                tracker.reset_interaction()
                tracker.baseline_frame = self.capture_baseline_frame(frame, roi)
            elif current_time >= tracker.decision_window_end:
                event = InteractionEvent.PICKED
                tracker.add_event(event)
                logger.info(f"[{tracker.zone_id}] EVENT: PICKED - environment still changed after {decision_window_sec}s ({change_magnitude:.1f}%)")
                self._send_event_to_api(
                    tracker.zone_id,
                    "PICKED",
                    tracker.get_neglect_rate(self.get_store_open_seconds()),
                    None
                )
                tracker.reset_interaction()
                tracker.baseline_frame = self.capture_baseline_frame(frame, roi)
        
        return event
    
    # Legacy function kept for backward compatibility
    def process_zone_state(
        self,
        tracker: ZoneTracker,
        hand_in_zone: bool,
        fg_mask: np.ndarray,
        current_time: float,
        tripwire_crossing: Optional[str] = None,
        hand_position: Optional[Tuple[float, float]] = None
    ) -> Optional[InteractionEvent]:
        """Legacy state machine - redirects to V4."""
        # Get the frame from the fg_mask context (this is a workaround)
        # In the main loop, we'll call process_zone_state_v4 directly
        return None
    
    # =========================================================================
    # TELEMETRY OUTPUT
    # =========================================================================
    
    def _send_event_to_api(self, zone_id: str, event_type: str, neglect_rate: float, 
                           picked_object: Optional[DetectedObject] = None) -> None:
        """
        Phase 4/5 Glue: Send individual event to the dashboard API.
        
        Phase 5 (preferred): Uses offline-capable SQLite queue that persists
        events locally and syncs to cloud when connectivity is available.
        
        Phase 4 (fallback): Direct HTTP POST to backend.
        
        Args:
            zone_id: The zone identifier
            event_type: "PICKED" or "REJECTED"
            neglect_rate: Current neglect rate percentage
            picked_object: Optional detected object that was picked
        """
        if not self.enable_api_telemetry:
            return
        
        timestamp = time.time()
        object_name = picked_object.class_name if picked_object else None
        
        # Print for debugging
        obj_str = f", object={object_name}" if object_name else ""
        print(f"Telemetry Generated: zone={zone_id}, event={event_type}{obj_str}, neglect={neglect_rate:.1f}%")
        
        # Phase 5: Use offline-capable queue (preferred)
        if self.telemetry_queue is not None:
            try:
                self.telemetry_queue.enqueue(
                    zone_id=zone_id,
                    event_type=event_type,
                    neglect_rate_pct=round(neglect_rate, 1),
                    timestamp=timestamp
                )
                logger.debug(f"Event queued: {zone_id}/{event_type}")
            except Exception as e:
                logger.error(f"Failed to queue event: {e}")
            return
        
        # Phase 4 fallback: Direct HTTP POST
        if not REQUESTS_AVAILABLE:
            return
        
        payload = {
            "timestamp": timestamp,
            "zone_id": zone_id,
            "event": event_type,
            "neglect_rate_pct": round(neglect_rate, 1)
        }
        
        try:
            requests.post(
                f"{self.api_url}/api/telemetry",
                json=payload,
                timeout=2
            )
        except Exception as e:
            logger.warning(f"Could not connect to dashboard API: {e}")
    
    def generate_telemetry(self) -> TelemetryPayload:
        """
        Generate JSON telemetry payload with current state of all zones.
        
        Returns:
            TelemetryPayload object
        """
        store_open_seconds = self.get_store_open_seconds()
        store_status = "OPEN" if self.is_store_open() else "CLOSED"
        
        zones_data = []
        for zone_id, tracker in self.zone_trackers.items():
            zone_data = {
                "zone_id": zone_id,
                "state": tracker.state.name,
                "idle_time_seconds": round(tracker.get_idle_time(), 1),
                "neglect_rate_percent": round(tracker.get_neglect_rate(store_open_seconds), 2),
                "total_taken": tracker.total_taken,
                "total_put_back": tracker.total_put_back,
                "total_touches": tracker.total_touches,
                "recent_events": list(tracker.recent_events)[-3:]  # Last 3 events
            }
            zones_data.append(zone_data)
        
        return TelemetryPayload(
            timestamp=datetime.now().isoformat(),
            store_status=store_status,
            zones=zones_data
        )
    
    def output_telemetry(self) -> None:
        """Print telemetry JSON to terminal and send to API."""
        telemetry = self.generate_telemetry()
        
        # Print to terminal
        print("\n" + "="*60)
        print("STORESENSE TELEMETRY")
        print("="*60)
        print(telemetry.to_json())
        print("="*60 + "\n")
        
        # Send to Phase 4 API if enabled
        if self.telemetry_sender is not None:
            try:
                self.telemetry_sender.send_telemetry_payload(telemetry.to_dict())
                logger.debug("Telemetry sent to API")
            except Exception as e:
                logger.warning(f"Failed to send telemetry to API: {e}")
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    # Tripwire visualization colors
    COLOR_TRIPWIRE = (255, 0, 255)          # Magenta - tripwire line
    COLOR_TRIPWIRE_ACTIVE = (0, 255, 255)   # Yellow - tripwire when hand in shelf
    COLOR_DECISION_WINDOW = (0, 165, 255)   # Orange - during decision window
    COLOR_TRAIL = (0, 255, 128)             # Green - hand trail line
    COLOR_OBJECT_PICKED = (255, 128, 0)     # Cyan - picked object indicator
    
    def draw_visualization(
        self,
        frame: np.ndarray,
        hand_boxes: List[Tuple[int, int, int, int]],
        fg_mask: np.ndarray
    ) -> np.ndarray:
        """
        Draw visualization overlay on frame.
        
        Shows:
        - Hand bounding boxes
        - Hand trail line connecting to changed region
        - Zone labels with PICKED/REJECTED metrics
        - Decision window countdown
        - Changed region marker
        - Store status
        """
        display = frame.copy()
        
        # Draw ROI info and hand trails for each zone
        for zone_id, tracker in self.zone_trackers.items():
            roi = tracker.roi
            x1, y1, x2, y2 = roi.get_bounds()
            
            # Color based on state
            if tracker.state == ZoneState.HAND_IN_ZONE:
                color = self.COLOR_HAND_PRESENT
            elif tracker.state == ZoneState.DECISION_WINDOW:
                color = self.COLOR_DECISION_WINDOW
            else:
                color = self.COLOR_IDLE
            
            # Draw hand trail if there are points
            if len(tracker.hand_trail) > 1:
                trail_points = list(tracker.hand_trail)
                for i in range(1, len(trail_points)):
                    # Gradient alpha for trail effect
                    alpha = int(255 * (i / len(trail_points)))
                    pt1 = (int(trail_points[i-1][0]), int(trail_points[i-1][1]))
                    pt2 = (int(trail_points[i][0]), int(trail_points[i][1]))
                    cv2.line(display, pt1, pt2, self.COLOR_TRAIL, 2)
            
            # Draw trail line to changed region if in decision window
            if tracker.state == ZoneState.DECISION_WINDOW and tracker.picked_object:
                obj = tracker.picked_object
                obj_center = obj.center
                
                # Draw line from last hand position to object center
                if tracker.last_hand_position:
                    hand_pt = (int(tracker.last_hand_position[0]), int(tracker.last_hand_position[1]))
                    cv2.line(display, hand_pt, obj_center, self.COLOR_OBJECT_PICKED, 3)
                    
                    # Draw a small circle at object position
                    cv2.circle(display, obj_center, 8, self.COLOR_OBJECT_PICKED, -1)
                    
                    # Draw scene change label
                    label = f"scene changed ({obj.confidence:.0%})"
                    cv2.putText(display, label, (obj_center[0] + 10, obj_center[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_OBJECT_PICKED, 2)
            
            # Draw label background
            label = f"{zone_id}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display, (x1, y1 - lh - 10), (x1 + lw + 4, y1), color, -1)
            cv2.putText(display, label, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Draw state indicator
            state_text = tracker.state.name
            if tracker.state == ZoneState.DECISION_WINDOW:
                # Show countdown
                remaining = max(0, tracker.decision_window_end - time.time())
                state_text = f"DECISION: {remaining:.1f}s"
                if tracker.object_change_detected:
                    state_text += f" [scene changed {tracker.change_magnitude:.1f}%]"
                if tracker.hand_returned_during_window:
                    state_text += " (RETURNED!)"
            cv2.putText(display, state_text, (x1, y1 - lh - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw metrics below ROI
            idle_time = tracker.get_idle_time()
            metrics = f"Idle:{idle_time:.0f}s | PICK:{tracker.total_picked} REJ:{tracker.total_rejected}"
            cv2.putText(display, metrics, (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw hand boxes
        for hx, hy, hw, hh in hand_boxes:
            cv2.rectangle(display, (hx, hy), (hx + hw, hy + hh),
                         self.COLOR_HAND_BOX, 2)
            cv2.putText(display, "HAND", (hx, hy - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_HAND_BOX, 2)
        
        # Draw store status
        status = "STORE: OPEN" if self.is_store_open() else "STORE: CLOSED (Paused)"
        status_color = (0, 255, 0) if self.is_store_open() else (0, 0, 255)
        cv2.putText(display, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Draw current time
        time_str = datetime.now().strftime("%H:%M:%S")
        cv2.putText(display, time_str, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw mode indicator
        cv2.putText(display, "Hand + Environment Change Detection", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        return display
    
    # =========================================================================
    # MAIN PROCESSING LOOP
    # =========================================================================
    
    def run(self) -> None:
        """
        Main processing loop.
        
        This is the core async loop that:
        1. Checks store hours and pauses if closed
        2. Reads frames from video stream
        3. Detects hands using MediaPipe
        4. Processes MOG2 with pause mechanism
        5. Updates zone state machines
        6. Outputs telemetry periodically
        7. Shows visualization (optional)
        """
        logger.info("\n" + "="*60)
        logger.info("STARTING STORESENSE ENGINE")
        logger.info("="*60)
        
        self.is_running = True
        self.engine_start_time = time.time()
        self.last_telemetry_time = time.time()
        
        # Initialize zone timestamps
        for tracker in self.zone_trackers.values():
            tracker.last_touched_timestamp = time.time()
        
        try:
            while self.is_running:
                current_time = time.time()
                
                # ---------------------------------------------------------
                # Time-Aware Check: Pause if store is closed
                # ---------------------------------------------------------
                if not self.is_store_open():
                    if not self.is_paused:
                        logger.info("Store is CLOSED - pausing processing")
                        self.is_paused = True
                    
                    # Show paused state if display is enabled
                    if self.show_display:
                        ret, frame = self.read_frame()
                        if ret:
                            cv2.putText(frame, "STORE CLOSED - PAUSED", 
                                       (50, self.frame_height // 2),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                            cv2.imshow("StoreSense Engine", frame)
                    
                    # Sleep to save CPU
                    key = cv2.waitKey(1000) & 0xFF
                    if key == ord('q'):
                        break
                    continue
                else:
                    if self.is_paused:
                        logger.info("Store is OPEN - resuming processing")
                        self.is_paused = False
                
                # ---------------------------------------------------------
                # Read Frame
                # ---------------------------------------------------------
                ret, frame = self.read_frame()
                if not ret:
                    logger.warning("Failed to read frame, retrying...")
                    time.sleep(0.1)
                    continue
                
                # ---------------------------------------------------------
                # Detect Hands
                # ---------------------------------------------------------
                hand_boxes = self.detect_hands(frame)
                
                # ---------------------------------------------------------
                # Process MOG2 with Pause Mechanism
                # ---------------------------------------------------------
                fg_mask = self.process_mog2(frame, hand_boxes)

                # MediaPipe can miss hands in difficult lighting / angles.
                # Use motion contours as a fallback so ROI interactions still trigger.
                motion_hand_boxes = self.detect_hands_from_motion(fg_mask, frame.shape)
                if hand_boxes:
                    hand_boxes.extend(
                        box for box in motion_hand_boxes
                        if not any(
                            abs(box[0] - hx) < 20 and abs(box[1] - hy) < 20
                            for hx, hy, _, _ in hand_boxes
                        )
                    )
                else:
                    hand_boxes = motion_hand_boxes
                
                # ---------------------------------------------------------
                # Process Each Zone's State Machine (V4 ROI Boundary + Objects)
                # ---------------------------------------------------------
                for zone_id, tracker in self.zone_trackers.items():
                    # Check if any hand is in this zone's ROI boundary
                    hand_in_zone = False
                    hand_position = None
                    
                    for hand_box in hand_boxes:
                        if self.is_hand_in_roi_boundary(hand_box, tracker.roi):
                            hand_in_zone = True
                            hand_position = self.get_hand_center(hand_box)
                            break
                    
                    # Process V4 state machine with ROI boundary detection
                    event = self.process_zone_state_v4(
                        tracker, hand_in_zone, frame, current_time,
                        hand_position=hand_position
                    )
                
                # ---------------------------------------------------------
                # Periodic Telemetry Output
                # ---------------------------------------------------------
                if current_time - self.last_telemetry_time >= self.telemetry_interval:
                    self.output_telemetry()
                    self.last_telemetry_time = current_time
                
                # ---------------------------------------------------------
                # Visualization
                # ---------------------------------------------------------
                if self.show_display:
                    display = self.draw_visualization(frame, hand_boxes, fg_mask)
                    
                    # Also show fg_mask in small window
                    fg_display = cv2.resize(fg_mask, (320, 240))
                    cv2.imshow("StoreSense Engine", display)
                    cv2.imshow("MOG2 Mask", fg_display)
                
                # ---------------------------------------------------------
                # Handle Keyboard Input
                # ---------------------------------------------------------
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested")
                    break
                elif key == ord('t'):
                    # Manual telemetry output
                    self.output_telemetry()
                elif key == ord('r'):
                    # Reset all zone trackers
                    for tracker in self.zone_trackers.values():
                        tracker.reset_interaction()
                        tracker.total_taken = 0
                        tracker.total_put_back = 0
                        tracker.total_touches = 0
                        tracker.recent_events.clear()
                    logger.info("All zone trackers reset")
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            self.stop()
    
    def stop(self) -> None:
        """Stop the engine and release resources."""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
        
        if self.hand_tracker:
            self.hand_tracker.close()
        
        cv2.destroyAllWindows()
        
        # Final telemetry
        logger.info("\n" + "="*60)
        logger.info("FINAL TELEMETRY")
        logger.info("="*60)
        self.output_telemetry()
        
        logger.info("StoreSense Engine stopped")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def parse_arguments():
    """Parse command line arguments for production deployment."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='StoreSense Edge-AI Vision Engine',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config', '-c',
        default='config.json',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run without display window (for systemd/server deployment)'
    )
    
    parser.add_argument(
        '--telemetry-interval', '-t',
        type=float,
        default=10.0,
        help='Seconds between telemetry outputs'
    )
    
    parser.add_argument(
        '--api-url',
        default='http://localhost:3001',
        help='Backend API URL for telemetry'
    )
    
    parser.add_argument(
        '--no-api',
        action='store_true',
        help='Disable API telemetry (local-only mode)'
    )
    
    parser.add_argument(
        '--no-queue',
        action='store_true',
        help='Disable offline queue, use direct HTTP instead'
    )
    
    parser.add_argument(
        '--queue-db',
        default='telemetry_queue.db',
        help='Path for offline queue SQLite database'
    )
    
    parser.add_argument(
        '--phone', '-p',
        type=str,
        metavar='IP_ADDRESS',
        help='Use phone camera via IP Webcam app (e.g., --phone 192.168.1.100)'
    )
    
    parser.add_argument(
        '--phone-port',
        type=int,
        default=8080,
        help='IP Webcam port (default: 8080)'
    )

    parser.add_argument(
        '--yolo-model',
        default='yolov8n.pt',
        help='Path to YOLO weights file (use your custom shuttlecock model here)'
    )
    
    return parser.parse_args()


def get_phone_camera_url(ip_address: str, port: int = 8080) -> str:
    """
    Generate the video stream URL for IP Webcam app.
    
    IP Webcam app (Android) provides multiple stream formats:
    - /video - MJPEG stream (most compatible)
    - /videofeed - Alternative MJPEG
    - /shot.jpg - Single frame capture
    
    Args:
        ip_address: Phone's IP address on local network
        port: IP Webcam port (default 8080)
        
    Returns:
        URL string for the video stream
    """
    return f"http://{ip_address}:{port}/video"


def main():
    """Main entry point for StoreSense Phase 2 Engine."""
    
    # Parse command line arguments
    args = parse_arguments()
    
    print("\n" + "="*60)
    print("  STORESENSE - Phase 4: Vision Loop with Object Detection")
    print("="*60 + "\n")
    
    # Configuration from arguments
    config_path = args.config
    telemetry_interval = args.telemetry_interval
    show_display = not args.headless
    api_url = args.api_url
    enable_api = not args.no_api
    use_queue = not args.no_queue
    queue_db = args.queue_db
    yolo_model_path = args.yolo_model
    
    if args.headless:
        print("Running in HEADLESS mode (no display)")
    
    # Handle phone camera option
    phone_url = None
    if args.phone:
        phone_url = get_phone_camera_url(args.phone, args.phone_port)
        print(f"Phone camera mode enabled")
        print(f"  - IP Address: {args.phone}")
        print(f"  - Port: {args.phone_port}")
        print(f"  - Stream URL: {phone_url}")
        print("\nMake sure IP Webcam app is running on your phone!")
        print("Download from: https://play.google.com/store/apps/details?id=com.pas.webcam")
        print("-" * 60)
    
    # Create engine
    engine = StoreSenseEngine(
        config_path=config_path,
        telemetry_interval=telemetry_interval,
        show_display=show_display,
        api_url=api_url,
        enable_api_telemetry=enable_api,
        use_offline_queue=use_queue,
        queue_db_path=queue_db,
        yolo_model_path=yolo_model_path
    )
    
    # Load configuration from Phase 1
    print("[Step 1/3] Loading configuration...")
    if not engine.load_config():
        print("ERROR: Failed to load config.json")
        print("Please run Phase 1 calibration first.")
        return
    
    # Override RTSP URL if phone camera specified
    if phone_url:
        engine.rtsp_url = phone_url
        print(f"[Phone Camera] Overriding video source: {phone_url}")
    
    # Connect to video stream
    print("[Step 2/3] Connecting to video stream...")
    if not engine.connect():
        print("ERROR: Failed to connect to video stream")
        if args.phone:
            print("\nPhone camera troubleshooting:")
            print("  1. Make sure IP Webcam app is running and 'Start server' is pressed")
            print("  2. Verify your phone and computer are on the same WiFi network")
            print(f"  3. Try opening {phone_url} in your browser to test")
            print("  4. Check if firewall is blocking the connection")
        return
    
    # Initialize models
    print("[Step 3/3] Initializing models...")
    if not engine.initialize_models():
        print("ERROR: Failed to initialize models")
        return
    
    # Run the engine
    print("\n" + "="*60)
    print("ENGINE READY")
    print("="*60)
    print("Controls:")
    print("  - Press 'Q' to quit")
    print("  - Press 'T' for manual telemetry output")
    print("  - Press 'R' to reset all counters")
    if args.phone:
        print(f"\nPhone Camera: {args.phone}:{args.phone_port}")
    print("="*60 + "\n")
    
    engine.run()


if __name__ == "__main__":
    main()
