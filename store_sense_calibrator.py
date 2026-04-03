"""
StoreSense Phase 1: Environment Setup & Camera Calibration (V3 Divider)
=========================================================================

This module handles the complete Phase 1 setup for StoreSense:
1. Global settings collection (store hours, interaction friction window)
2. RTSP/HTTP stream ingestion with robust retry logic (supports IP Webcam apps)
3. Interactive ROI (Region of Interest) calibration via mouse
4. Vertical divider placement within each ROI (shelf edge detection)
5. Zone labeling and persistent storage to structured config.json
6. MediaPipe Hands and MOG2 Background Subtractor initialization

V3 Divider Upgrade:
- After drawing each ROI rectangle, user positions a vertical divider inside it
- The divider can be dragged with the mouse or nudged with keyboard controls
- User marks whether the shelf is on the left or right side of the divider
- The saved config includes both divider coordinates and shelf side

Supported Camera Sources:
- RTSP streams: rtsp://ip:port/stream
- HTTP/MJPEG streams: http://ip:port/video (IP Webcam app)
- Local webcam: integer index (0, 1, etc.)

Architecture Note:
- NO object detection models (YOLO, etc.) are used to save compute power
- Relies entirely on MediaPipe (hand tracking) + OpenCV MOG2 (pixel change detection)

Author: StoreSense Team
Version: 3.0 (Divider)
"""

import cv2
import json
import time
import logging
import re
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Union

# Configure logging for debugging and monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES - Structured Configuration Payload
# =============================================================================

@dataclass
class GlobalSettings:
    """
    Global store settings that apply to all zones.
    
    Attributes:
        store_open_time: Store opening time in HH:MM format (e.g., "08:00")
        store_close_time: Store closing time in HH:MM format (e.g., "22:00")
        interaction_friction_window: Minimum seconds a hand must remain in a zone
                                     before it's considered a valid interaction.
                                     This filters out quick pass-throughs.
                                     Typical values: 7, 10, or 15 seconds.
    """
    store_open_time: str
    store_close_time: str
    interaction_friction_window: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "store_open_time": self.store_open_time,
            "store_close_time": self.store_close_time,
            "interaction_friction_window": self.interaction_friction_window
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'GlobalSettings':
        """Create GlobalSettings from dictionary."""
        return cls(
            store_open_time=data["store_open_time"],
            store_close_time=data["store_close_time"],
            interaction_friction_window=data["interaction_friction_window"]
        )
    
    def is_store_open(self, current_time: str) -> bool:
        """
        Check if the store is currently open.
        
        Args:
            current_time: Current time in HH:MM format
            
        Returns:
            bool: True if store is open
        """
        return self.store_open_time <= current_time <= self.store_close_time


@dataclass
class ROI:
    """
    Represents a single Region of Interest (shelf zone) with optional divider.
    
    Attributes:
        zone_id: User-defined identifier (e.g., "Shelf_1_Spices")
        x: Top-left X coordinate
        y: Top-left Y coordinate
        width: Width of the bounding box
        height: Height of the bounding box
        tripwire: Optional vertical divider line as [(x1,y1), (x2,y2)]
        shelf_side: Which side of the divider is the shelf ("left" or "right")
    """
    zone_id: str
    x: int
    y: int
    width: int
    height: int
    tripwire: Optional[List[Tuple[int, int]]] = None
    shelf_side: str = "right"
    
    def to_dict(self) -> dict:
        """Convert ROI to dictionary for JSON serialization."""
        data = {
            "zone_id": self.zone_id,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height
        }
        if self.tripwire:
            data["tripwire"] = self.tripwire
        data["shelf_side"] = self.shelf_side
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ROI':
        """Create ROI from dictionary."""
        tripwire = data.get("tripwire")
        if tripwire:
            # Convert list of lists to list of tuples
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
    
    def contains_point(self, px: int, py: int) -> bool:
        """Check if a point (px, py) is inside this ROI."""
        return (self.x <= px <= self.x + self.width and 
                self.y <= py <= self.y + self.height)
    
    def get_tripwire_normal(self) -> Optional[Tuple[float, float]]:
        """
        Get the outward-pointing normal vector of the tripwire.
        
        The normal points from the shelf side toward the customer side.
        
        Returns:
            Tuple (nx, ny) normalized vector, or None if no tripwire defined.
        """
        if not self.tripwire or len(self.tripwire) != 2:
            return None
        
        if self.shelf_side == "left":
            return (1.0, 0.0)
        return (-1.0, 0.0)


@dataclass
class Configuration:
    """
    Complete StoreSense configuration payload.
    
    This structured dataclass contains all configuration data needed
    for the StoreSense system, including global settings and ROI definitions.
    
    Attributes:
        version: Configuration schema version for future compatibility
        rtsp_url: The RTSP stream URL or camera index
        calibration_timestamp: When the calibration was performed
        global_settings: Store-wide settings (hours, friction window)
        rois: List of defined shelf zones/regions of interest
    """
    version: str
    rtsp_url: str
    calibration_timestamp: str
    global_settings: GlobalSettings
    rois: List[ROI] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert entire configuration to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "rtsp_url": self.rtsp_url,
            "calibration_timestamp": self.calibration_timestamp,
            "global_settings": self.global_settings.to_dict(),
            "rois": [roi.to_dict() for roi in self.rois]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Configuration':
        """Create Configuration from dictionary."""
        return cls(
            version=data["version"],
            rtsp_url=data["rtsp_url"],
            calibration_timestamp=data["calibration_timestamp"],
            global_settings=GlobalSettings.from_dict(data["global_settings"]),
            rois=[ROI.from_dict(roi_data) for roi_data in data.get("rois", [])]
        )
    
    def save_to_file(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'Configuration':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class CalibrationState:
    """
    Tracks the current state of ROI and tripwire drawing during calibration.
    
    Attributes:
        is_drawing: True while mouse button is held down for ROI
        start_x: Starting X coordinate of current rectangle
        start_y: Starting Y coordinate of current rectangle
        current_x: Current X position during drag
        current_y: Current Y position during drag
        roi_completed: Flag set when a rectangle is finished
        
        # Tripwire drawing state
        tripwire_mode: True when drawing tripwire line for current ROI
        tripwire_point1: First click point (shelf side)
        tripwire_point2: Second click point (customer side)
        tripwire_completed: Flag set when tripwire line is finished
    """
    is_drawing: bool = False
    start_x: int = 0
    start_y: int = 0
    current_x: int = 0
    current_y: int = 0
    roi_completed: bool = False
    
    # Divider/tripwire editing
    tripwire_mode: bool = False
    tripwire_point1: Optional[Tuple[int, int]] = None
    tripwire_point2: Optional[Tuple[int, int]] = None
    tripwire_completed: bool = False
    divider_x: int = 0
    dragging_divider: bool = False
    shelf_side: str = "right"
    
    # Pending ROI (waiting for tripwire)
    pending_roi_x: int = 0
    pending_roi_y: int = 0
    pending_roi_width: int = 0
    pending_roi_height: int = 0


# =============================================================================
# MAIN CALIBRATOR CLASS
# =============================================================================

class StoreSenseCalibrator:
    """
    Encapsulates all Phase 1 functionality for StoreSense calibration.
    
    This class manages:
    - Global settings collection via terminal prompts
    - RTSP stream connection with robust retry logic
    - Interactive ROI drawing interface via OpenCV mouse callbacks
    - Configuration persistence to structured JSON
    - Initialization of tracking models (MediaPipe Hands, MOG2)
    
    Architecture Note:
        No object detection models (YOLO, etc.) are used. The system relies
        entirely on MediaPipe for hand tracking and OpenCV MOG2 for detecting
        pixel changes in shelf zones after a hand leaves.
    
    Usage:
        calibrator = StoreSenseCalibrator(rtsp_url=0)
        calibrator.collect_global_settings()
        calibrator.connect()
        calibrator.capture_calibration_frame()
        calibrator.run_calibration()
        calibrator.initialize_models()
        calibrator.release()
    """
    
    # Class constants
    WINDOW_NAME = "StoreSense Calibrator - Draw ROIs & Divider"
    DEFAULT_CONFIG_PATH = "config.json"
    CONFIG_VERSION = "3.0"
    
    # Colors for drawing (BGR format for OpenCV)
    COLOR_DRAWING = (0, 255, 255)      # Yellow - while actively drawing
    COLOR_CONFIRMED = (0, 255, 0)       # Green - confirmed/saved ROIs
    COLOR_TEXT = (255, 255, 255)        # White - text labels
    COLOR_INSTRUCTIONS = (0, 200, 255)  # Orange - instruction overlay
    COLOR_SETTINGS = (255, 200, 0)      # Cyan - settings display
    COLOR_TRIPWIRE = (255, 0, 255)      # Magenta - tripwire line
    COLOR_TRIPWIRE_SHELF = (255, 100, 100)   # Light blue - shelf side point
    COLOR_TRIPWIRE_CUSTOMER = (100, 100, 255) # Light red - customer side point
    
    def __init__(
        self,
        rtsp_url: Union[str, int],
        config_path: str = DEFAULT_CONFIG_PATH,
        max_retries: int = 5,
        retry_delay: float = 2.0,
        connection_timeout: float = 10.0
    ):
        """
        Initialize the StoreSense Calibrator.
        
        Args:
            rtsp_url: Video source, can be:
                      - RTSP stream: "rtsp://192.168.1.100:554/stream"
                      - HTTP/MJPEG stream: "http://192.168.1.100:8080/video" (IP Webcam app)
                      - Webcam index: 0, 1, etc.
            config_path: Path to save/load ROI configuration JSON
            max_retries: Maximum number of connection retry attempts
            retry_delay: Seconds to wait between retry attempts
            connection_timeout: Timeout in seconds for initial connection
        """
        # Stream configuration
        self.rtsp_url: Union[str, int] = rtsp_url
        self.config_path = Path(config_path)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.connection_timeout = connection_timeout
        
        # Video capture object (initialized on connect)
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Frame storage for calibration
        self.calibration_frame: Optional[np.ndarray] = None
        self.display_frame: Optional[np.ndarray] = None
        
        # Global settings (collected via terminal prompts)
        self.global_settings: Optional[GlobalSettings] = None
        
        # ROI management
        self.rois: List[ROI] = []
        self.drawing_state = CalibrationState()
        
        # Model references (initialized in initialize_models())
        self.hand_tracker = None         # MediaPipe Hands instance
        self.bg_subtractor = None        # MOG2 Background Subtractor instance
        
        logger.info(f"StoreSenseCalibrator initialized for stream: {rtsp_url}")
    
    # =========================================================================
    # GLOBAL SETTINGS COLLECTION
    # =========================================================================
    
    def collect_global_settings(self) -> GlobalSettings:
        """
        Collect global store settings via terminal input prompts.
        
        Prompts the user for:
        1. Store Open Time (HH:MM format)
        2. Store Close Time (HH:MM format)
        3. Interaction Friction Window (seconds)
        
        The friction window defines how long a hand must remain in a zone
        before it's considered a genuine interaction (vs. a pass-through).
        
        Returns:
            GlobalSettings: The collected settings dataclass
        """
        print("\n" + "="*60)
        print("  GLOBAL STORE SETTINGS")
        print("="*60)
        print("Please enter the following settings for your store.\n")
        
        # -----------------------------------------------------------------
        # Collect Store Open Time
        # -----------------------------------------------------------------
        while True:
            store_open_time = input("Store Open Time (HH:MM format, e.g., '08:00'): ").strip()
            if self._validate_time_format(store_open_time):
                break
            print("  Invalid format. Please use HH:MM (e.g., '08:00', '09:30')")
        
        # -----------------------------------------------------------------
        # Collect Store Close Time
        # -----------------------------------------------------------------
        while True:
            store_close_time = input("Store Close Time (HH:MM format, e.g., '22:00'): ").strip()
            if self._validate_time_format(store_close_time):
                if store_close_time > store_open_time:
                    break
                print("  Close time must be after open time.")
            else:
                print("  Invalid format. Please use HH:MM (e.g., '21:00', '22:30')")
        
        # -----------------------------------------------------------------
        # Collect Interaction Friction Window
        # -----------------------------------------------------------------
        print("\nInteraction Friction Window:")
        print("  This is the minimum time (in seconds) a hand must stay in a zone")
        print("  before it's considered a valid interaction. Typical values: 7, 10, 15")
        
        while True:
            friction_input = input("Friction Window in seconds (e.g., '7', '10', '15'): ").strip()
            try:
                friction_window = int(friction_input)
                if 1 <= friction_window <= 60:
                    break
                print("  Please enter a value between 1 and 60 seconds.")
            except ValueError:
                print("  Invalid input. Please enter a number (e.g., 7, 10, 15)")
        
        # Create and store the GlobalSettings
        self.global_settings = GlobalSettings(
            store_open_time=store_open_time,
            store_close_time=store_close_time,
            interaction_friction_window=friction_window
        )
        
        # Display confirmation
        print("\n" + "-"*40)
        print("Settings confirmed:")
        print(f"  - Store Hours: {store_open_time} to {store_close_time}")
        print(f"  - Friction Window: {friction_window} seconds")
        print("-"*40 + "\n")
        
        logger.info(f"Global settings collected: {store_open_time}-{store_close_time}, "
                   f"friction={friction_window}s")
        
        return self.global_settings
    
    def _validate_time_format(self, time_str: str) -> bool:
        """
        Validate that a string is in HH:MM format.
        
        Args:
            time_str: The time string to validate
            
        Returns:
            bool: True if valid HH:MM format
        """
        # Regex pattern for HH:MM (00:00 to 23:59)
        pattern = r'^([01]?[0-9]|2[0-3]):([0-5][0-9])$'
        if re.match(pattern, time_str):
            return True
        return False
    
    # =========================================================================
    # RTSP CONNECTION METHODS
    # =========================================================================
    
    def connect(self) -> bool:
        """
        Establish connection to video stream with robust retry logic.
        
        Supports multiple stream types:
        - Local webcam (integer index)
        - RTSP streams (rtsp://...)
        - HTTP/MJPEG streams (http://...) - e.g., IP Webcam app
        
        Implements a retry mechanism to handle network instability common
        with network streams.
        
        Returns:
            bool: True if connection successful, False otherwise.
        """
        for attempt in range(1, self.max_retries + 1):
            logger.info(f"Connection attempt {attempt}/{self.max_retries}...")
            
            try:
                # Determine stream type and create VideoCapture with appropriate backend
                if isinstance(self.rtsp_url, int):
                    # Local webcam - use DirectShow on Windows for better compatibility
                    logger.info(f"Connecting to local webcam (index {self.rtsp_url})...")
                    self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_DSHOW)
                    
                elif isinstance(self.rtsp_url, str):
                    url_lower = self.rtsp_url.lower()
                    
                    if url_lower.startswith('http://') or url_lower.startswith('https://'):
                        # HTTP/MJPEG stream (IP Webcam app, etc.)
                        logger.info(f"Connecting to HTTP stream: {self.rtsp_url}")
                        # Use default backend for HTTP streams (handles MJPEG)
                        self.cap = cv2.VideoCapture(self.rtsp_url)
                        
                    elif url_lower.startswith('rtsp://'):
                        # RTSP stream - use FFMPEG backend
                        logger.info(f"Connecting to RTSP stream: {self.rtsp_url}")
                        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                        
                    else:
                        # Try as generic URL or file path
                        logger.info(f"Connecting to: {self.rtsp_url}")
                        self.cap = cv2.VideoCapture(self.rtsp_url)
                else:
                    logger.error(f"Invalid stream source type: {type(self.rtsp_url)}")
                    return False
                
                # Optimize for low latency by minimizing buffer size
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Set connection timeout for network streams
                if isinstance(self.rtsp_url, str):
                    self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 
                               int(self.connection_timeout * 1000))
                
                # Verify connection by reading a test frame
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        logger.info("Successfully connected to video stream!")
                        logger.info(f"Frame dimensions: {frame.shape[1]}x{frame.shape[0]}")
                        return True
                    else:
                        logger.warning("Stream opened but failed to read frame")
                else:
                    logger.warning("Failed to open video capture")
                    
            except Exception as e:
                logger.error(f"Connection error: {e}")
            
            # Clean up failed connection attempt
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
            # Wait before next retry (skip delay on last attempt)
            if attempt < self.max_retries:
                logger.info(f"Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
        
        logger.error(f"Failed to connect after {self.max_retries} attempts")
        return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the RTSP stream with error handling.
        
        Implements retry logic for dropped frames or temporary disconnections,
        which are common with RTSP streams over unreliable networks.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success_flag, frame or None)
        """
        # Ensure we have a valid connection
        if self.cap is None or not self.cap.isOpened():
            logger.warning("Capture not initialized, attempting reconnect...")
            if not self.connect():
                return False, None
        
        # Attempt to read frame with retries for transient failures
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        while consecutive_failures < max_consecutive_failures:
            ret, frame = self.cap.read()
            
            if ret and frame is not None:
                return True, frame
            
            consecutive_failures += 1
            logger.warning(f"Frame read failed ({consecutive_failures}/{max_consecutive_failures})")
            
            # Brief pause before retry
            time.sleep(0.1)
            
            # After several failures, attempt full reconnection
            if consecutive_failures >= 5:
                logger.info("Multiple failures detected, attempting full reconnection...")
                self.cap.release()
                if not self.connect():
                    return False, None
                consecutive_failures = 0
        
        logger.error("Exceeded maximum consecutive frame failures")
        return False, None
    
    def capture_calibration_frame(self) -> bool:
        """
        Capture and store a clear frame for ROI calibration.
        
        Skips initial frames to ensure camera has stabilized and
        auto-exposure has adjusted.
        
        Returns:
            bool: True if frame captured successfully.
        """
        logger.info("Capturing calibration frame...")
        
        # Skip initial frames to allow camera to stabilize
        for _ in range(10):
            self.read_frame()
        
        # Capture the actual calibration frame
        ret, frame = self.read_frame()
        if ret and frame is not None:
            self.calibration_frame = frame.copy()
            self.display_frame = frame.copy()
            logger.info("Calibration frame captured successfully")
            logger.info(f"Frame dimensions: {frame.shape[1]}x{frame.shape[0]}")
            return True
        
        logger.error("Failed to capture calibration frame")
        return False
    
    # =========================================================================
    # ROI CALIBRATION METHODS
    # =========================================================================
    
    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        """
        OpenCV mouse callback for interactive ROI and tripwire drawing.
        
        Two-phase drawing:
        1. ROI Phase: Click-and-drag to draw rectangle
        2. Tripwire Phase: Click two points to define the shelf edge line
        
        Tripwire convention:
        - First click = shelf side (inside/product side)
        - Second click = customer side (outside)
        - Hand crossing from first→second triggers interaction
        
        Args:
            event: OpenCV mouse event type (cv2.EVENT_*)
            x: Mouse X coordinate
            y: Mouse Y coordinate
            flags: OpenCV event flags (unused)
            param: Additional callback parameters (unused)
        """
        state = self.drawing_state
        
        # =====================================================================
        # DIVIDER MODE: Slide the vertical divider inside the ROI
        # =====================================================================
        if state.tripwire_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                roi_x1 = state.pending_roi_x
                roi_x2 = state.pending_roi_x + state.pending_roi_width
                roi_y1 = state.pending_roi_y
                roi_y2 = state.pending_roi_y + state.pending_roi_height
                if roi_x1 <= x <= roi_x2 and roi_y1 <= y <= roi_y2:
                    state.divider_x = max(roi_x1 + 5, min(roi_x2 - 5, x))
                    state.dragging_divider = True
                    state.tripwire_point1 = (state.divider_x, roi_y1)
                    state.tripwire_point2 = (state.divider_x, roi_y2)
                self._update_display()
            elif event == cv2.EVENT_MOUSEMOVE and state.dragging_divider:
                roi_x1 = state.pending_roi_x
                roi_x2 = state.pending_roi_x + state.pending_roi_width
                state.divider_x = max(roi_x1 + 5, min(roi_x2 - 5, x))
                state.tripwire_point1 = (state.divider_x, state.pending_roi_y)
                state.tripwire_point2 = (state.divider_x, state.pending_roi_y + state.pending_roi_height)
                self._update_display()
            elif event == cv2.EVENT_LBUTTONUP:
                state.dragging_divider = False
            return
        
        # =====================================================================
        # ROI MODE: Click-and-drag to draw rectangle
        # =====================================================================
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing a new rectangle
            state.is_drawing = True
            state.start_x = x
            state.start_y = y
            state.current_x = x
            state.current_y = y
            state.roi_completed = False
            
        elif event == cv2.EVENT_MOUSEMOVE:
            # Update rectangle dimensions while dragging
            if state.is_drawing:
                state.current_x = x
                state.current_y = y
                self._update_display()
                
        elif event == cv2.EVENT_LBUTTONUP:
            # Complete the rectangle
            if state.is_drawing:
                state.is_drawing = False
                state.current_x = x
                state.current_y = y
                
                # Normalize coordinates to ensure positive width/height
                x1 = min(state.start_x, state.current_x)
                y1 = min(state.start_y, state.current_y)
                x2 = max(state.start_x, state.current_x)
                y2 = max(state.start_y, state.current_y)
                
                width = x2 - x1
                height = y2 - y1
                
                # Only accept rectangles with meaningful size
                if width > 10 and height > 10:
                    # Store pending ROI and switch to divider mode
                    state.pending_roi_x = x1
                    state.pending_roi_y = y1
                    state.pending_roi_width = width
                    state.pending_roi_height = height
                    state.start_x = x1
                    state.start_y = y1
                    state.current_x = x2
                    state.current_y = y2
                    
                    # Enter divider mode with a centered vertical divider
                    state.tripwire_mode = True
                    state.shelf_side = "right"
                    state.divider_x = x1 + width // 2
                    state.tripwire_point1 = (state.divider_x, y1)
                    state.tripwire_point2 = (state.divider_x, y2)
                    state.tripwire_completed = False
                    state.dragging_divider = False
                    
                    print("\n" + "-"*40)
                    print("ROI rectangle drawn! Adjust the vertical divider inside the ROI.")
                    print("Controls:")
                    print("  - Drag divider with mouse, or use LEFT/RIGHT arrows / A/D")
                    print("  - Press 'L' if shelf is on left side of divider")
                    print("  - Press 'R' if shelf is on right side of divider")
                    print("  - Press ENTER to confirm divider and enter zone ID")
                    print("-"*40)
                    
                    self._update_display()
                else:
                    logger.warning("Rectangle too small (min 10x10 pixels), ignoring")
    
    def _update_display(self) -> None:
        """
        Update the display frame with current ROIs, tripwires, drawing state, and settings.
        
        Redraws all confirmed ROIs (green) with their tripwire lines (magenta),
        the current rectangle being drawn (yellow), and instruction overlay.
        """
        if self.calibration_frame is None:
            return
        
        # Start with a fresh copy of the calibration frame
        self.display_frame = self.calibration_frame.copy()
        
        # Draw all confirmed ROIs in green with labels and tripwires
        for roi in self.rois:
            # Draw ROI rectangle
            cv2.rectangle(
                self.display_frame,
                (roi.x, roi.y),
                (roi.x + roi.width, roi.y + roi.height),
                self.COLOR_CONFIRMED,
                2
            )
            # Add zone label above the rectangle
            cv2.putText(
                self.display_frame,
                roi.zone_id,
                (roi.x, roi.y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.COLOR_TEXT,
                2
            )
            
            # Draw tripwire/divider if defined
            if roi.tripwire and len(roi.tripwire) == 2:
                p1, p2 = roi.tripwire
                cv2.line(self.display_frame, p1, p2, self.COLOR_TRIPWIRE, 2)
                mid_x = (p1[0] + p2[0]) // 2
                mid_y = (p1[1] + p2[1]) // 2
                normal = roi.get_tripwire_normal()
                if normal:
                    arrow_end = (int(mid_x + normal[0] * 20), int(mid_y + normal[1] * 20))
                    cv2.arrowedLine(self.display_frame, (mid_x, mid_y), arrow_end, 
                                   self.COLOR_TRIPWIRE, 2, tipLength=0.4)
                cv2.putText(
                    self.display_frame,
                    f"SHELF {roi.shelf_side.upper()}",
                    (mid_x + 8, max(20, mid_y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    self.COLOR_TRIPWIRE,
                    1
                )
        
        # Draw current rectangle being drawn (yellow while dragging)
        state = self.drawing_state
        if state.is_drawing or state.tripwire_mode:
            x1 = min(state.start_x, state.current_x) if not state.tripwire_mode else state.pending_roi_x
            y1 = min(state.start_y, state.current_y) if not state.tripwire_mode else state.pending_roi_y
            x2 = max(state.start_x, state.current_x) if not state.tripwire_mode else state.pending_roi_x + state.pending_roi_width
            y2 = max(state.start_y, state.current_y) if not state.tripwire_mode else state.pending_roi_y + state.pending_roi_height
            
            color = self.COLOR_DRAWING if state.is_drawing else self.COLOR_CONFIRMED
            cv2.rectangle(self.display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw divider in progress
            if state.tripwire_mode:
                if state.tripwire_point1 and state.tripwire_point2:
                    cv2.line(self.display_frame, state.tripwire_point1,
                            state.tripwire_point2, self.COLOR_TRIPWIRE, 3)
                    label_x = state.tripwire_point1[0] + 10
                    label_y = max(20, state.pending_roi_y + 20)
                    cv2.putText(
                        self.display_frame,
                        f"SHELF {state.shelf_side.upper()} | ENTER=confirm",
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        self.COLOR_TRIPWIRE,
                        1
                    )
        
        # Add global settings display in top-right corner
        if self.global_settings:
            settings_text = [
                f"Store: {self.global_settings.store_open_time} - {self.global_settings.store_close_time}",
                f"Friction: {self.global_settings.interaction_friction_window}s"
            ]
            frame_width = self.display_frame.shape[1]
            y_offset = 30
            for text in settings_text:
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                x_pos = frame_width - text_size[0] - 10
                cv2.putText(
                    self.display_frame, text, (x_pos, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_SETTINGS, 1
                )
                y_offset += 20
        
        # Add instruction overlay in top-left corner
        if state.tripwire_mode:
            instructions = [
                "DIVIDER MODE:",
                "- Drag divider with mouse",
                "- LEFT/RIGHT or A/D to slide",
                "- L = shelf on left, R = shelf on right",
                "- ENTER = confirm divider",
                "- Press 'C' to cancel this ROI",
                f"- ROIs defined: {len(self.rois)}"
            ]
        else:
            instructions = [
                "Instructions:",
                "- Click and drag to draw ROI",
                "- Then place vertical divider",
                "- Press 'S' to save and continue",
                "- Press 'R' to reset all ROIs",
                "- Press 'Q' to quit without saving",
                f"- ROIs defined: {len(self.rois)}"
            ]
        
        y_offset = 30
        for text in instructions:
            cv2.putText(
                self.display_frame, text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_INSTRUCTIONS, 1
            )
            y_offset += 20
    
    def run_calibration(self) -> bool:
        """
        Run the interactive ROI and tripwire calibration interface.
        
        Opens an OpenCV window displaying the calibration frame. User can:
        - Click and drag to draw ROI rectangles
        - Click two points to define tripwire line within each ROI
        - Enter zone IDs via terminal after each tripwire is defined
        - Press 'S' to save configuration and continue
        - Press 'R' to reset/clear all ROIs
        - Press 'C' to cancel current ROI/tripwire drawing
        - Press 'Q' to quit without saving
        
        Returns:
            bool: True if calibration completed and saved successfully.
        """
        if self.calibration_frame is None:
            logger.error("No calibration frame. Call capture_calibration_frame() first.")
            return False
        
        if self.global_settings is None:
            logger.error("No global settings. Call collect_global_settings() first.")
            return False
        
        # Create OpenCV window and register mouse callback
        cv2.namedWindow(self.WINDOW_NAME)
        cv2.setMouseCallback(self.WINDOW_NAME, self._mouse_callback)
        
        # Print calibration instructions
        logger.info("\n" + "="*60)
        logger.info("ROI & DIVIDER CALIBRATION MODE (V3)")
        logger.info("="*60)
        logger.info("1. Draw rectangles around shelf zones using your mouse")
        logger.info("2. Slide the vertical divider inside each ROI")
        logger.info("3. Choose whether the shelf is on the LEFT or RIGHT side")
        logger.info("3. Enter a zone ID in this terminal")
        logger.info("Press ENTER to confirm divider, 'S' to save, 'R' to reset, 'C' to cancel, 'Q' to quit")
        logger.info("="*60 + "\n")
        
        self._update_display()
        
        # Main calibration loop
        while True:
            cv2.imshow(self.WINDOW_NAME, self.display_frame)
            key = cv2.waitKey(50) & 0xFF
            
            # Confirm divider placement
            state = self.drawing_state
            if state.tripwire_mode:
                if key in (81, 2424832, ord('a'), ord('A')):
                    state.divider_x = max(state.pending_roi_x + 5, state.divider_x - 5)
                elif key in (83, 2555904, ord('d'), ord('D')):
                    state.divider_x = min(state.pending_roi_x + state.pending_roi_width - 5, state.divider_x + 5)
                elif key == ord('l') or key == ord('L'):
                    state.shelf_side = "left"
                elif key == ord('r') or key == ord('R'):
                    state.shelf_side = "right"
                elif key in (13, 10):
                    state.tripwire_completed = True

                if state.tripwire_mode and state.tripwire_point1 and state.tripwire_point2:
                    state.tripwire_point1 = (state.divider_x, state.pending_roi_y)
                    state.tripwire_point2 = (state.divider_x, state.pending_roi_y + state.pending_roi_height)
                    self._update_display()

            if state.tripwire_mode and state.tripwire_completed:
                # Prompt for zone ID
                zone_id = input("\nEnter Zone ID for this ROI (e.g., 'Shelf_1_Spices'): ").strip()
                
                if zone_id:
                    # Create ROI with tripwire
                    assert state.tripwire_point1 is not None and state.tripwire_point2 is not None
                    tripwire = [state.tripwire_point1, state.tripwire_point2]
                    new_roi = ROI(
                        zone_id=zone_id,
                        x=state.pending_roi_x,
                        y=state.pending_roi_y,
                        width=state.pending_roi_width,
                        height=state.pending_roi_height,
                        tripwire=tripwire,
                        shelf_side=state.shelf_side
                    )
                    self.rois.append(new_roi)
                    logger.info(f"Added ROI: {zone_id}")
                    logger.info(f"  Box: ({new_roi.x}, {new_roi.y}, {new_roi.width}x{new_roi.height})")
                    logger.info(f"  Tripwire: {tripwire[0]} -> {tripwire[1]}")
                    print(f"\n  ROI '{zone_id}' saved with tripwire!")
                    print(f"  Divider X: {state.divider_x} | Shelf side: {state.shelf_side}")
                    print("  Draw another ROI or press 'S' to save.\n")
                else:
                    logger.warning("Empty zone ID entered, ROI discarded")
                    print("  ROI discarded (no zone ID entered)")
                
                # Reset drawing state
                state.tripwire_mode = False
                state.tripwire_point1 = None
                state.tripwire_point2 = None
                state.tripwire_completed = False
                state.dragging_divider = False
                state.roi_completed = False
                
                self._update_display()
            
            # Handle keyboard commands
            if key == ord('s') or key == ord('S'):
                if self.rois:
                    self.save_config()
                    logger.info("Configuration saved successfully!")
                    break
                else:
                    logger.warning("No ROIs defined. Draw at least one ROI before saving.")
                    print("No ROIs defined yet. Draw at least one ROI and tripwire.")
                    
            elif key == ord('r') or key == ord('R'):
                self.rois.clear()
                # Reset drawing state
                state.tripwire_mode = False
                state.tripwire_point1 = None
                state.tripwire_point2 = None
                state.tripwire_completed = False
                state.dragging_divider = False
                state.roi_completed = False
                logger.info("All ROIs cleared")
                print("All ROIs cleared. Start drawing again.")
                self._update_display()
                
            elif key == ord('c') or key == ord('C'):
                # Cancel current ROI/tripwire drawing
                if state.tripwire_mode:
                    state.tripwire_mode = False
                    state.tripwire_point1 = None
                    state.tripwire_point2 = None
                    state.tripwire_completed = False
                    state.dragging_divider = False
                    logger.info("Current ROI cancelled")
                    print("Current ROI cancelled. Draw a new one.")
                    self._update_display()
                
            elif key == ord('q') or key == ord('Q'):
                logger.info("Calibration cancelled by user")
                cv2.destroyAllWindows()
                return False
        
        cv2.destroyAllWindows()
        return True
    
    # =========================================================================
    # CONFIGURATION PERSISTENCE
    # =========================================================================
    
    def save_config(self) -> None:
        """
        Save complete configuration (global settings + ROIs) to JSON file.
        
        Creates a structured Configuration object and saves it to the
        specified config file path.
        """
        if self.global_settings is None:
            raise ValueError("Global settings not collected. Call collect_global_settings() first.")
        
        # Create the complete configuration object
        config = Configuration(
            version=self.CONFIG_VERSION,
            rtsp_url=str(self.rtsp_url),
            calibration_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            global_settings=self.global_settings,
            rois=self.rois
        )
        
        # Save to file
        config.save_to_file(str(self.config_path))
        
        logger.info(f"Configuration saved to {self.config_path}")
        logger.info(f"  - Global settings: {self.global_settings.store_open_time} - "
                   f"{self.global_settings.store_close_time}, "
                   f"friction={self.global_settings.interaction_friction_window}s")
        logger.info(f"  - Total ROIs saved: {len(self.rois)}")
    
    def load_config(self) -> bool:
        """
        Load configuration from JSON file.
        
        Returns:
            bool: True if configuration loaded successfully.
        """
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return False
        
        try:
            config = Configuration.load_from_file(str(self.config_path))
            self.global_settings = config.global_settings
            self.rois = config.rois
            
            logger.info(f"Loaded configuration from {self.config_path}")
            logger.info(f"  - Version: {config.version}")
            logger.info(f"  - Store hours: {self.global_settings.store_open_time} - "
                       f"{self.global_settings.store_close_time}")
            logger.info(f"  - Friction window: {self.global_settings.interaction_friction_window}s")
            logger.info(f"  - ROIs: {len(self.rois)}")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return False
    
    # =========================================================================
    # MODEL INITIALIZATION
    # =========================================================================
    
    def initialize_models(self) -> bool:
        """
        Initialize MediaPipe Hands and MOG2 Background Subtractor.
        
        This initializes the two core components for hand tracking and
        pixel change detection:
        
        1. MediaPipe Hands: For detecting and tracking hands in the frame
        2. MOG2 Background Subtractor: For detecting pixel changes in shelf
           zones after a hand leaves (to determine if item was taken/returned)
        
        Note: NO object detection models (YOLO, etc.) are used to save compute.
        
        Returns:
            bool: True if all models initialized successfully.
        """
        logger.info("\n" + "="*60)
        logger.info("INITIALIZING TRACKING MODELS")
        logger.info("="*60)
        
        success = True
        
        # -----------------------------------------------------------------
        # Initialize MediaPipe Hands
        # -----------------------------------------------------------------
        try:
            import mediapipe as mp
            
            # Access the hands solution from MediaPipe
            mp_hands = mp.solutions.hands
            
            # Initialize with parameters optimized for real-time tracking
            # static_image_mode=False enables temporal smoothing between frames
            # max_num_hands=2 allows tracking both hands simultaneously
            # min_detection_confidence=0.7 balances accuracy vs speed
            # min_tracking_confidence=0.5 allows tracking through brief occlusions
            self.hand_tracker = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            
            logger.info("[SUCCESS] MediaPipe Hands initialized")
            logger.info("         - Mode: Video (temporal smoothing enabled)")
            logger.info("         - Max hands: 2")
            logger.info("         - Detection confidence: 0.7")
            logger.info("         - Tracking confidence: 0.5")
            
        except ImportError:
            logger.error("[FAILED] MediaPipe not installed")
            logger.error("         Run: pip install mediapipe")
            success = False
        except Exception as e:
            logger.error(f"[FAILED] MediaPipe initialization error: {e}")
            success = False
        
        # -----------------------------------------------------------------
        # Initialize MOG2 Background Subtractor
        # -----------------------------------------------------------------
        try:
            # Create MOG2 Background Subtractor with specified parameters:
            # - history=500: Number of frames used to build background model
            # - varThreshold=50: Threshold for foreground detection sensitivity
            # - detectShadows=True: Enable shadow detection for better accuracy
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500,
                varThreshold=50,
                detectShadows=True
            )
            
            logger.info("[SUCCESS] MOG2 Background Subtractor initialized")
            logger.info("         - History: 500 frames")
            logger.info("         - Variance Threshold: 50")
            logger.info("         - Shadow Detection: Enabled")
            
        except Exception as e:
            logger.error(f"[FAILED] MOG2 initialization error: {e}")
            success = False
        
        # -----------------------------------------------------------------
        # Print Summary
        # -----------------------------------------------------------------
        if success:
            logger.info("="*60)
            logger.info("ALL MODELS INITIALIZED SUCCESSFULLY!")
            logger.info("StoreSense is ready for Phase 2: Hand Tracking")
            logger.info("="*60 + "\n")
        else:
            logger.error("="*60)
            logger.error("MODEL INITIALIZATION FAILED")
            logger.error("Please check error messages and install missing dependencies.")
            logger.error("="*60 + "\n")
        
        return success
    
    # =========================================================================
    # CLEANUP
    # =========================================================================
    
    def release(self) -> None:
        """
        Release all resources and clean up.
        
        Should be called when calibration is complete or on error
        to properly release video capture and close windows.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        if self.hand_tracker is not None:
            self.hand_tracker.close()
            self.hand_tracker = None
        
        cv2.destroyAllWindows()
        logger.info("Resources released")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main entry point for StoreSense Phase 1 calibration.
    
    Executes the complete Phase 1 workflow:
    1. Collect global store settings (hours, friction window)
    2. Connect to RTSP/video stream
    3. Capture calibration frame
    4. Run interactive ROI calibration
    5. Save structured configuration to JSON
    6. Initialize tracking models
    """
    
    # =========================================================================
    # CONFIGURATION - Modify these values for your setup
    # =========================================================================
    
    # RTSP stream URL - replace with your camera's RTSP address
    # Common RTSP URL formats:
    #   - rtsp://username:password@ip:port/stream
    #   - rtsp://ip:554/live/ch00_0
    #
    # For testing with local webcam, use integer index:
    #   - 0 for default webcam
    #   - 1 for secondary camera, etc.
    
    RTSP_URL = 0  # Default to laptop camera for now
    # RTSP_URL = "rtsp://admin:password@192.168.1.100:554/stream"
    
    # Configuration file path
    CONFIG_PATH = "config.json"
    
    # =========================================================================
    # EXECUTION
    # =========================================================================
    
    print("\n" + "="*60)
    print("  STORESENSE - Phase 1: Environment Setup & Calibration")
    print("="*60)
    print("\nThis wizard will guide you through:")
    print("  1. Global store settings configuration")
    print("  2. Video stream connection")
    print("  3. Interactive ROI (shelf zone) definition")
    print("  4. Tracking model initialization")
    print("="*60 + "\n")
    
    # Create calibrator instance
    calibrator = StoreSenseCalibrator(
        rtsp_url=RTSP_URL,
        config_path=CONFIG_PATH,
        max_retries=5,
        retry_delay=2.0,
        connection_timeout=10.0
    )
    
    try:
        # Step 1: Collect global store settings
        print("[Step 1/5] Collecting global store settings...")
        calibrator.collect_global_settings()
        
        # Step 2: Connect to video stream
        print("[Step 2/5] Connecting to video stream...")
        if not calibrator.connect():
            print("ERROR: Failed to connect to video stream")
            print("Please check your RTSP_URL and network connection")
            return
        
        # Step 3: Capture calibration frame
        print("[Step 3/5] Capturing calibration frame...")
        if not calibrator.capture_calibration_frame():
            print("ERROR: Failed to capture calibration frame")
            return
        
        # Step 4: Run interactive ROI calibration
        print("[Step 4/5] Starting ROI calibration interface...")
        print("          (A window will open - draw ROIs on the shelf image)")
        print("          Press 'S' to save, 'R' to reset, 'Q' to quit")
        if not calibrator.run_calibration():
            print("Calibration was cancelled")
            return
        
        # Step 5: Initialize tracking models
        print("[Step 5/5] Initializing tracking models...")
        if not calibrator.initialize_models():
            print("WARNING: Some models failed to initialize")
            print("Please install missing dependencies and try again")
            return
        
        # Success - Print completion summary
        print("\n" + "="*60)
        print("  PHASE 1 COMPLETE!")
        print("="*60)
        print(f"  Configuration saved to: {CONFIG_PATH}")
        print(f"  ")
        print(f"  Global Settings:")
        print(f"    - Store Hours: {calibrator.global_settings.store_open_time} - "
              f"{calibrator.global_settings.store_close_time}")
        print(f"    - Friction Window: {calibrator.global_settings.interaction_friction_window}s")
        print(f"  ")
        print(f"  ROI Zones Defined: {len(calibrator.rois)}")
        for roi in calibrator.rois:
            print(f"    - {roi.zone_id}: ({roi.x}, {roi.y}, {roi.width}x{roi.height})")
        print(f"  ")
        print(f"  Models Ready:")
        print(f"    - MediaPipe Hands: Initialized")
        print(f"    - MOG2 Background Subtractor: Initialized")
        print(f"  ")
        print("  You can now proceed to Phase 2: Hand Tracking")
        print("="*60 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user (Ctrl+C)")
        
    finally:
        # Always clean up resources
        calibrator.release()


if __name__ == "__main__":
    main()
