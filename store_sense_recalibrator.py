"""
StoreSense Phase 3: Maintenance & Recalibration Mode
=====================================================

This module provides quick maintenance capabilities when the camera gets
bumped or shifted. Instead of redoing the entire Phase 1 setup, users can
simply drag existing ROI boxes to realign them with the shifted video feed.

Key Features:
1. Loads existing config.json preserving ALL global_settings
2. Displays live/captured frame with existing ROIs overlaid
3. Click-and-drag to translate (move) entire ROI boxes
4. Click near edges/corners to resize ROIs
5. Safe JSON handling - never loses store hours or zone names

The Problem This Solves:
In physical stores, cameras get bumped by ladders, cleaning crews, or
curious customers. When this happens, the digital ROI coordinates no
longer align with physical shelves. This tool provides a 2-minute fix
instead of a 20-minute complete recalibration.

Author: StoreSense Team
Version: 3.0
"""

import cv2
import json
import time
import copy
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Union
from enum import Enum, auto

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class DragMode(Enum):
    """Defines what type of drag operation is being performed."""
    NONE = auto()           # Not dragging
    TRANSLATE = auto()      # Moving the entire box
    RESIZE_TL = auto()      # Resizing from top-left corner
    RESIZE_TR = auto()      # Resizing from top-right corner
    RESIZE_BL = auto()      # Resizing from bottom-left corner
    RESIZE_BR = auto()      # Resizing from bottom-right corner
    RESIZE_T = auto()       # Resizing from top edge
    RESIZE_B = auto()       # Resizing from bottom edge
    RESIZE_L = auto()       # Resizing from left edge
    RESIZE_R = auto()       # Resizing from right edge


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ROI:
    """
    Represents a single Region of Interest (shelf zone).
    
    Stores both current position and original position for reset functionality.
    """
    zone_id: str
    x: int
    y: int
    width: int
    height: int
    
    # Original values for reset
    original_x: int = 0
    original_y: int = 0
    original_width: int = 0
    original_height: int = 0
    
    def __post_init__(self):
        """Store original values on creation."""
        self.original_x = self.x
        self.original_y = self.y
        self.original_width = self.width
        self.original_height = self.height
    
    def reset(self) -> None:
        """Reset to original position."""
        self.x = self.original_x
        self.y = self.original_y
        self.width = self.original_width
        self.height = self.original_height
    
    def get_bounds(self) -> Tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) bounds."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    def contains_point(self, px: int, py: int) -> bool:
        """Check if point is inside this ROI."""
        x1, y1, x2, y2 = self.get_bounds()
        return x1 <= px <= x2 and y1 <= py <= y2
    
    def get_edge_zone(self, px: int, py: int, threshold: int = 15) -> DragMode:
        """
        Determine which edge/corner/center the point is near.
        
        Args:
            px, py: Point coordinates
            threshold: Pixel distance to consider "near" an edge
            
        Returns:
            DragMode indicating the drag operation type
        """
        x1, y1, x2, y2 = self.get_bounds()
        
        # Check if point is inside the ROI
        if not self.contains_point(px, py):
            return DragMode.NONE
        
        near_left = abs(px - x1) <= threshold
        near_right = abs(px - x2) <= threshold
        near_top = abs(py - y1) <= threshold
        near_bottom = abs(py - y2) <= threshold
        
        # Corner detection (highest priority)
        if near_top and near_left:
            return DragMode.RESIZE_TL
        if near_top and near_right:
            return DragMode.RESIZE_TR
        if near_bottom and near_left:
            return DragMode.RESIZE_BL
        if near_bottom and near_right:
            return DragMode.RESIZE_BR
        
        # Edge detection
        if near_top:
            return DragMode.RESIZE_T
        if near_bottom:
            return DragMode.RESIZE_B
        if near_left:
            return DragMode.RESIZE_L
        if near_right:
            return DragMode.RESIZE_R
        
        # Center - translate mode
        return DragMode.TRANSLATE
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON (excludes original values)."""
        return {
            "zone_id": self.zone_id,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ROI':
        """Create ROI from dictionary."""
        return cls(
            zone_id=data["zone_id"],
            x=data["x"],
            y=data["y"],
            width=data["width"],
            height=data["height"]
        )


@dataclass
class DragState:
    """
    Tracks the current drag operation state.
    
    Stores all information needed for smooth dragging/resizing.
    """
    is_dragging: bool = False
    mode: DragMode = DragMode.NONE
    active_roi_index: int = -1      # Index of ROI being dragged
    
    # Mouse position tracking
    start_mouse_x: int = 0          # Where drag started
    start_mouse_y: int = 0
    current_mouse_x: int = 0
    current_mouse_y: int = 0
    
    # ROI position at drag start (for calculating deltas)
    start_roi_x: int = 0
    start_roi_y: int = 0
    start_roi_width: int = 0
    start_roi_height: int = 0
    
    def reset(self) -> None:
        """Reset drag state."""
        self.is_dragging = False
        self.mode = DragMode.NONE
        self.active_roi_index = -1


# =============================================================================
# MAIN RECALIBRATOR CLASS
# =============================================================================

class StoreSenseRecalibrator:
    """
    Maintenance tool for realigning ROIs when camera position shifts.
    
    This class provides:
    - Safe loading of existing config.json (preserves global_settings)
    - RTSP connection with retry logic
    - Interactive drag-and-drop for ROI translation
    - Edge/corner resizing for fine adjustments
    - Safe save with backup of original config
    
    Usage:
        recalibrator = StoreSenseRecalibrator()
        recalibrator.load_config()
        recalibrator.connect()
        recalibrator.capture_frame()
        recalibrator.run_recalibration()
    """
    
    WINDOW_NAME = "StoreSense Recalibrator - Drag to Realign"
    
    # Colors (BGR)
    COLOR_ROI_NORMAL = (0, 255, 0)       # Green - normal ROI
    COLOR_ROI_ACTIVE = (0, 255, 255)     # Yellow - being dragged
    COLOR_ROI_MODIFIED = (255, 165, 0)   # Orange - modified from original
    COLOR_HANDLE = (255, 0, 255)         # Magenta - resize handles
    COLOR_TEXT = (255, 255, 255)         # White - labels
    COLOR_INSTRUCTIONS = (0, 200, 255)   # Orange - instructions
    COLOR_STATUS = (0, 255, 0)           # Green - status text
    
    # Resize handle size
    HANDLE_SIZE = 8
    EDGE_THRESHOLD = 15  # Pixels from edge to trigger resize mode
    
    def __init__(
        self,
        config_path: str = "config.json",
        max_retries: int = 5,
        retry_delay: float = 2.0
    ):
        """
        Initialize the Recalibrator.
        
        Args:
            config_path: Path to existing config.json
            max_retries: Max connection retry attempts
            retry_delay: Seconds between retries
        """
        self.config_path = Path(config_path)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Full config (preserved exactly)
        self.full_config: Dict[str, Any] = {}
        
        # ROIs (editable)
        self.rois: List[ROI] = []
        
        # Video capture
        self.cap: Optional[cv2.VideoCapture] = None
        self.rtsp_url: Optional[Union[str, int]] = None
        
        # Frame storage
        self.calibration_frame: Optional[np.ndarray] = None
        self.display_frame: Optional[np.ndarray] = None
        self.frame_width: int = 0
        self.frame_height: int = 0
        
        # Drag state
        self.drag_state = DragState()
        
        # Track if any modifications were made
        self.has_modifications: bool = False
        
        logger.info("StoreSenseRecalibrator initialized")
    
    # =========================================================================
    # CONFIGURATION LOADING (SAFE PRESERVATION)
    # =========================================================================
    
    def load_config(self) -> bool:
        """
        Load existing config.json, preserving ALL data.
        
        This method is critical for safety - it keeps a complete copy
        of the config so we never lose global_settings or other data.
        
        Returns:
            bool: True if config loaded successfully
        """
        if not self.config_path.exists():
            logger.error(f"Config file not found: {self.config_path}")
            logger.error("Please run Phase 1 calibration first to create config.json")
            return False
        
        try:
            with open(self.config_path, 'r') as f:
                self.full_config = json.load(f)
            
            # Extract RTSP URL
            self.rtsp_url = self.full_config.get("rtsp_url", 0)
            
            # Load ROIs
            rois_data = self.full_config.get("rois", [])
            self.rois = [ROI.from_dict(roi_data) for roi_data in rois_data]
            
            # Log what we loaded (without modifying)
            global_settings = self.full_config.get("global_settings", {})
            
            logger.info("="*50)
            logger.info("CONFIG LOADED SUCCESSFULLY")
            logger.info("="*50)
            logger.info(f"Config version: {self.full_config.get('version', 'N/A')}")
            logger.info(f"RTSP URL: {self.rtsp_url}")
            logger.info(f"ROIs loaded: {len(self.rois)}")
            for roi in self.rois:
                logger.info(f"  - {roi.zone_id}: ({roi.x}, {roi.y}, {roi.width}x{roi.height})")
            
            if global_settings:
                logger.info("Global Settings (PRESERVED):")
                logger.info(f"  - Store hours: {global_settings.get('store_open_time', 'N/A')} - "
                          f"{global_settings.get('store_close_time', 'N/A')}")
                logger.info(f"  - Friction window: {global_settings.get('interaction_friction_window', 'N/A')}s")
            
            logger.info("="*50)
            
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return False
    
    # =========================================================================
    # RTSP CONNECTION
    # =========================================================================
    
    def connect(self) -> bool:
        """
        Connect to RTSP stream with retry logic.
        
        Returns:
            bool: True if connected successfully
        """
        for attempt in range(1, self.max_retries + 1):
            logger.info(f"Connection attempt {attempt}/{self.max_retries}...")
            
            try:
                if isinstance(self.rtsp_url, int) or (isinstance(self.rtsp_url, str) and self.rtsp_url.isdigit()):
                    url = int(self.rtsp_url) if isinstance(self.rtsp_url, str) else self.rtsp_url
                    self.cap = cv2.VideoCapture(url, cv2.CAP_DSHOW)
                else:
                    self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self.frame_height, self.frame_width = frame.shape[:2]
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
    
    def capture_frame(self) -> bool:
        """
        Capture a stable frame for recalibration.
        
        Returns:
            bool: True if frame captured
        """
        if self.cap is None:
            return False
        
        # Skip initial frames for camera stabilization
        for _ in range(10):
            self.cap.read()
        
        ret, frame = self.cap.read()
        if ret and frame is not None:
            self.calibration_frame = frame.copy()
            self.display_frame = frame.copy()
            logger.info("Calibration frame captured")
            return True
        
        logger.error("Failed to capture frame")
        return False
    
    # =========================================================================
    # MOUSE CALLBACK FOR DRAG & DROP
    # =========================================================================
    
    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        """
        Advanced mouse callback for drag-and-drop ROI manipulation.
        
        Handles:
        - Click inside ROI center → translate (move) entire box
        - Click near edge → resize in that direction
        - Click near corner → resize from corner
        
        The math ensures smooth, intuitive dragging by calculating
        delta from the original mouse position, not absolute coordinates.
        """
        state = self.drag_state
        
        # -----------------------------------------------------------------
        # MOUSE DOWN - Start drag operation
        # -----------------------------------------------------------------
        if event == cv2.EVENT_LBUTTONDOWN:
            # Find which ROI was clicked (if any)
            # Check in reverse order so top-most (last drawn) is selected first
            for i in range(len(self.rois) - 1, -1, -1):
                roi = self.rois[i]
                drag_mode = roi.get_edge_zone(x, y, self.EDGE_THRESHOLD)
                
                if drag_mode != DragMode.NONE:
                    # Start drag operation
                    state.is_dragging = True
                    state.mode = drag_mode
                    state.active_roi_index = i
                    state.start_mouse_x = x
                    state.start_mouse_y = y
                    state.current_mouse_x = x
                    state.current_mouse_y = y
                    
                    # Store ROI state at drag start
                    state.start_roi_x = roi.x
                    state.start_roi_y = roi.y
                    state.start_roi_width = roi.width
                    state.start_roi_height = roi.height
                    
                    logger.debug(f"Started {drag_mode.name} on {roi.zone_id}")
                    break
        
        # -----------------------------------------------------------------
        # MOUSE MOVE - Update drag
        # -----------------------------------------------------------------
        elif event == cv2.EVENT_MOUSEMOVE:
            state.current_mouse_x = x
            state.current_mouse_y = y
            
            if state.is_dragging and state.active_roi_index >= 0:
                roi = self.rois[state.active_roi_index]
                
                # Calculate delta from drag start
                dx = x - state.start_mouse_x
                dy = y - state.start_mouse_y
                
                # Apply transformation based on drag mode
                if state.mode == DragMode.TRANSLATE:
                    # Move entire box
                    roi.x = state.start_roi_x + dx
                    roi.y = state.start_roi_y + dy
                
                elif state.mode == DragMode.RESIZE_TL:
                    # Top-left corner: adjust x, y, width, height
                    roi.x = state.start_roi_x + dx
                    roi.y = state.start_roi_y + dy
                    roi.width = state.start_roi_width - dx
                    roi.height = state.start_roi_height - dy
                
                elif state.mode == DragMode.RESIZE_TR:
                    # Top-right corner: adjust y, width, height
                    roi.y = state.start_roi_y + dy
                    roi.width = state.start_roi_width + dx
                    roi.height = state.start_roi_height - dy
                
                elif state.mode == DragMode.RESIZE_BL:
                    # Bottom-left corner: adjust x, width, height
                    roi.x = state.start_roi_x + dx
                    roi.width = state.start_roi_width - dx
                    roi.height = state.start_roi_height + dy
                
                elif state.mode == DragMode.RESIZE_BR:
                    # Bottom-right corner: adjust width, height
                    roi.width = state.start_roi_width + dx
                    roi.height = state.start_roi_height + dy
                
                elif state.mode == DragMode.RESIZE_T:
                    # Top edge: adjust y, height
                    roi.y = state.start_roi_y + dy
                    roi.height = state.start_roi_height - dy
                
                elif state.mode == DragMode.RESIZE_B:
                    # Bottom edge: adjust height
                    roi.height = state.start_roi_height + dy
                
                elif state.mode == DragMode.RESIZE_L:
                    # Left edge: adjust x, width
                    roi.x = state.start_roi_x + dx
                    roi.width = state.start_roi_width - dx
                
                elif state.mode == DragMode.RESIZE_R:
                    # Right edge: adjust width
                    roi.width = state.start_roi_width + dx
                
                # Enforce minimum size
                if roi.width < 20:
                    roi.width = 20
                if roi.height < 20:
                    roi.height = 20
                
                # Clamp to frame bounds
                roi.x = max(0, min(roi.x, self.frame_width - roi.width))
                roi.y = max(0, min(roi.y, self.frame_height - roi.height))
                
                self.has_modifications = True
                self._update_display()
        
        # -----------------------------------------------------------------
        # MOUSE UP - End drag operation
        # -----------------------------------------------------------------
        elif event == cv2.EVENT_LBUTTONUP:
            if state.is_dragging:
                roi = self.rois[state.active_roi_index]
                logger.info(f"Moved {roi.zone_id} to ({roi.x}, {roi.y}, {roi.width}x{roi.height})")
                state.reset()
                self._update_display()
    
    # =========================================================================
    # DISPLAY RENDERING
    # =========================================================================
    
    def _update_display(self) -> None:
        """
        Update the display with ROIs, handles, and instructions.
        
        Shows:
        - All ROIs with their zone labels
        - Resize handles on corners and edges
        - Different colors for normal/active/modified states
        - Instructions overlay
        """
        if self.calibration_frame is None:
            return
        
        self.display_frame = self.calibration_frame.copy()
        state = self.drag_state
        
        # Draw each ROI
        for i, roi in enumerate(self.rois):
            x1, y1, x2, y2 = roi.get_bounds()
            
            # Determine color based on state
            is_active = (state.is_dragging and state.active_roi_index == i)
            is_modified = (roi.x != roi.original_x or roi.y != roi.original_y or
                          roi.width != roi.original_width or roi.height != roi.original_height)
            
            if is_active:
                color = self.COLOR_ROI_ACTIVE
                thickness = 3
            elif is_modified:
                color = self.COLOR_ROI_MODIFIED
                thickness = 2
            else:
                color = self.COLOR_ROI_NORMAL
                thickness = 2
            
            # Draw main rectangle
            cv2.rectangle(self.display_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw zone label
            label = roi.zone_id
            if is_modified:
                label += " *"  # Indicate modification
            
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(self.display_frame, (x1, y1 - lh - 10), (x1 + lw + 4, y1), color, -1)
            cv2.putText(self.display_frame, label, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Draw resize handles (small squares at corners and edges)
            handle_color = self.COLOR_HANDLE
            hs = self.HANDLE_SIZE
            
            # Corner handles
            corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
            for cx, cy in corners:
                cv2.rectangle(self.display_frame, 
                            (cx - hs//2, cy - hs//2), 
                            (cx + hs//2, cy + hs//2), 
                            handle_color, -1)
            
            # Edge handles (midpoints)
            edges = [
                ((x1 + x2)//2, y1),  # Top
                ((x1 + x2)//2, y2),  # Bottom
                (x1, (y1 + y2)//2),  # Left
                (x2, (y1 + y2)//2),  # Right
            ]
            for ex, ey in edges:
                cv2.rectangle(self.display_frame,
                            (ex - hs//2, ey - hs//2),
                            (ex + hs//2, ey + hs//2),
                            handle_color, -1)
            
            # Show original position as dotted outline if modified
            if is_modified and not is_active:
                ox1, oy1 = roi.original_x, roi.original_y
                ox2 = roi.original_x + roi.original_width
                oy2 = roi.original_y + roi.original_height
                # Draw dotted rectangle (using dashed line segments)
                self._draw_dotted_rect(ox1, oy1, ox2, oy2, (128, 128, 128))
        
        # Draw instructions
        instructions = [
            "RECALIBRATION MODE",
            "==================",
            "- Click & drag CENTER to MOVE box",
            "- Click & drag EDGE to resize",
            "- Click & drag CORNER to resize",
            "",
            "Press 'S' to SAVE changes",
            "Press 'R' to RESET to original",
            "Press 'Q' to QUIT without saving",
        ]
        
        y_offset = 30
        for text in instructions:
            cv2.putText(self.display_frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_INSTRUCTIONS, 1)
            y_offset += 20
        
        # Show modification status
        if self.has_modifications:
            status = "STATUS: MODIFIED (unsaved)"
            status_color = (0, 165, 255)  # Orange
        else:
            status = "STATUS: No changes"
            status_color = self.COLOR_STATUS
        
        cv2.putText(self.display_frame, status, 
                   (10, self.frame_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Show current drag mode
        if state.is_dragging:
            mode_text = f"Dragging: {state.mode.name}"
            cv2.putText(self.display_frame, mode_text,
                       (10, self.frame_height - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_ROI_ACTIVE, 1)
    
    def _draw_dotted_rect(self, x1: int, y1: int, x2: int, y2: int, 
                          color: Tuple[int, int, int], gap: int = 5) -> None:
        """Draw a dotted/dashed rectangle."""
        # Top edge
        for x in range(x1, x2, gap * 2):
            cv2.line(self.display_frame, (x, y1), (min(x + gap, x2), y1), color, 1)
        # Bottom edge
        for x in range(x1, x2, gap * 2):
            cv2.line(self.display_frame, (x, y2), (min(x + gap, x2), y2), color, 1)
        # Left edge
        for y in range(y1, y2, gap * 2):
            cv2.line(self.display_frame, (x1, y), (x1, min(y + gap, y2)), color, 1)
        # Right edge
        for y in range(y1, y2, gap * 2):
            cv2.line(self.display_frame, (x2, y), (x2, min(y + gap, y2)), color, 1)
    
    # =========================================================================
    # SAVE / RESET / QUIT
    # =========================================================================
    
    def save_config(self) -> bool:
        """
        Save updated config with SAFE handling.
        
        This method:
        1. Creates a backup of the original config
        2. Updates ONLY the ROI coordinates
        3. Preserves ALL other data (global_settings, version, etc.)
        4. Writes atomically to prevent corruption
        
        Returns:
            bool: True if saved successfully
        """
        try:
            # Create backup first
            backup_path = self.config_path.with_suffix('.json.backup')
            with open(self.config_path, 'r') as f:
                original_content = f.read()
            with open(backup_path, 'w') as f:
                f.write(original_content)
            logger.info(f"Backup created: {backup_path}")
            
            # Update ONLY the ROIs in the full config
            self.full_config["rois"] = [roi.to_dict() for roi in self.rois]
            
            # Update timestamp
            self.full_config["recalibration_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Write to temp file first (atomic write)
            temp_path = self.config_path.with_suffix('.json.tmp')
            with open(temp_path, 'w') as f:
                json.dump(self.full_config, f, indent=2)
            
            # Rename temp to actual (atomic on most systems)
            temp_path.replace(self.config_path)
            
            logger.info("="*50)
            logger.info("CONFIG SAVED SUCCESSFULLY")
            logger.info("="*50)
            logger.info(f"File: {self.config_path}")
            logger.info("Updated ROIs:")
            for roi in self.rois:
                logger.info(f"  - {roi.zone_id}: ({roi.x}, {roi.y}, {roi.width}x{roi.height})")
            
            # Verify global_settings preserved
            global_settings = self.full_config.get("global_settings", {})
            if global_settings:
                logger.info("Global Settings (PRESERVED):")
                logger.info(f"  - Store hours: {global_settings.get('store_open_time')} - "
                          f"{global_settings.get('store_close_time')}")
                logger.info(f"  - Friction window: {global_settings.get('interaction_friction_window')}s")
            logger.info("="*50)
            
            # Update original positions to current
            for roi in self.rois:
                roi.original_x = roi.x
                roi.original_y = roi.y
                roi.original_width = roi.width
                roi.original_height = roi.height
            
            self.has_modifications = False
            return True
            
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False
    
    def reset_all_rois(self) -> None:
        """Reset all ROIs to their original positions."""
        for roi in self.rois:
            roi.reset()
        
        self.has_modifications = False
        logger.info("All ROIs reset to original positions")
        self._update_display()
    
    # =========================================================================
    # MAIN RECALIBRATION LOOP
    # =========================================================================
    
    def run_recalibration(self) -> bool:
        """
        Run the interactive recalibration interface.
        
        Returns:
            bool: True if changes were saved
        """
        if self.calibration_frame is None:
            logger.error("No frame captured. Call capture_frame() first.")
            return False
        
        if not self.rois:
            logger.error("No ROIs loaded. Nothing to recalibrate.")
            return False
        
        # Create window and set mouse callback
        cv2.namedWindow(self.WINDOW_NAME)
        cv2.setMouseCallback(self.WINDOW_NAME, self._mouse_callback)
        
        logger.info("\n" + "="*60)
        logger.info("RECALIBRATION MODE ACTIVE")
        logger.info("="*60)
        logger.info("Drag ROI boxes to realign with the shifted camera view.")
        logger.info("- Drag center to move")
        logger.info("- Drag edges/corners to resize")
        logger.info("Press 'S' to save, 'R' to reset, 'Q' to quit")
        logger.info("="*60 + "\n")
        
        self._update_display()
        saved = False
        
        while True:
            cv2.imshow(self.WINDOW_NAME, self.display_frame)
            key = cv2.waitKey(50) & 0xFF
            
            if key == ord('s') or key == ord('S'):
                if self.has_modifications:
                    if self.save_config():
                        saved = True
                        print("\n[SUCCESS] Configuration saved!")
                else:
                    print("\n[INFO] No changes to save.")
                    
            elif key == ord('r') or key == ord('R'):
                self.reset_all_rois()
                print("\n[INFO] All ROIs reset to original positions.")
                
            elif key == ord('q') or key == ord('Q'):
                if self.has_modifications:
                    print("\n[WARNING] You have unsaved changes!")
                    print("Press 'Q' again to confirm quit, or 'S' to save first.")
                    confirm_key = cv2.waitKey(3000) & 0xFF
                    if confirm_key == ord('q') or confirm_key == ord('Q'):
                        print("[INFO] Exiting without saving.")
                        break
                    elif confirm_key == ord('s') or confirm_key == ord('S'):
                        self.save_config()
                        saved = True
                        break
                else:
                    break
        
        cv2.destroyAllWindows()
        return saved
    
    # =========================================================================
    # CLEANUP
    # =========================================================================
    
    def release(self) -> None:
        """Release resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Resources released")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for StoreSense Phase 3 Recalibration."""
    
    print("\n" + "="*60)
    print("  STORESENSE - Phase 3: Maintenance & Recalibration")
    print("="*60)
    print("\nThis tool allows you to realign ROI boxes after the camera")
    print("has been bumped or shifted, WITHOUT losing your store settings.")
    print("="*60 + "\n")
    
    CONFIG_PATH = "config.json"
    
    recalibrator = StoreSenseRecalibrator(
        config_path=CONFIG_PATH,
        max_retries=5,
        retry_delay=2.0
    )
    
    try:
        # Step 1: Load existing config
        print("[Step 1/3] Loading existing configuration...")
        if not recalibrator.load_config():
            print("ERROR: Failed to load config.json")
            print("Please run Phase 1 calibration first.")
            return
        
        # Step 2: Connect to video stream
        print("\n[Step 2/3] Connecting to video stream...")
        if not recalibrator.connect():
            print("ERROR: Failed to connect to video stream")
            return
        
        # Step 3: Capture frame
        print("[Step 3/3] Capturing calibration frame...")
        if not recalibrator.capture_frame():
            print("ERROR: Failed to capture frame")
            return
        
        # Run recalibration
        print("\n" + "="*60)
        print("RECALIBRATION INTERFACE")
        print("="*60)
        print("A window will open. Drag the ROI boxes to realign them.")
        print("- Press 'S' to SAVE changes")
        print("- Press 'R' to RESET boxes")  
        print("- Press 'Q' to QUIT")
        print("="*60 + "\n")
        
        saved = recalibrator.run_recalibration()
        
        # Summary
        print("\n" + "="*60)
        if saved:
            print("  RECALIBRATION COMPLETE - Changes Saved!")
        else:
            print("  RECALIBRATION ENDED - No Changes Saved")
        print("="*60 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        
    finally:
        recalibrator.release()


if __name__ == "__main__":
    main()
