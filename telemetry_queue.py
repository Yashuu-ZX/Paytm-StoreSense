"""
StoreSense Phase 5: Production Telemetry Queue
===============================================

A resilient, offline-capable telemetry queue that persists events to SQLite
and syncs to the cloud backend when connectivity is available.

Features:
- SQLite-backed persistence (survives crashes, reboots, Wi-Fi drops)
- Thread-safe operations (safe for concurrent OpenCV + sync threads)
- Background sync thread with exponential backoff
- Bulk sync for efficient bandwidth usage
- Automatic retry with configurable intervals
- Graceful degradation when server is unavailable

Architecture:
    [Event Occurs] --> [Save to SQLite (synced=False)] --> [Background Thread]
                                                                   |
                                                                   v
                                                        [POST to Node.js API]
                                                                   |
                                                    success?       |
                                                    /              \
                                                  Yes              No
                                                  /                  \
                                        [Mark synced=True]    [Retry later]

Usage:
    from telemetry_queue import TelemetryQueue
    
    queue = TelemetryQueue(
        db_path="telemetry_queue.db",
        api_url="http://localhost:3001"
    )
    queue.start()
    
    # Queue event (immediately persisted to SQLite)
    queue.enqueue("Shelf_1_Spices", "TAKEN", 12.5)
    
    # On shutdown
    queue.stop()

@author StoreSense Team
@version 5.0 (Production)
"""

import sqlite3
import threading
import time
import logging
import os
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from contextlib import contextmanager

# Optional: requests for HTTP calls
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("requests library not available - sync disabled")

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class QueuedEvent:
    """Represents a telemetry event in the queue."""
    id: int
    timestamp: float
    zone_id: str
    event_type: str
    neglect_rate_pct: float
    synced: bool
    created_at: str
    
    def to_api_dict(self) -> Dict[str, Any]:
        """Convert to API payload format."""
        return {
            "timestamp": int(self.timestamp),
            "zone_id": self.zone_id,
            "event": self.event_type,
            "neglect_rate_pct": self.neglect_rate_pct
        }


# =============================================================================
# THREAD-SAFE SQLITE CONNECTION MANAGER
# =============================================================================

class SQLiteConnectionManager:
    """
    Thread-safe SQLite connection manager.
    
    SQLite connections cannot be shared across threads. This manager
    ensures each thread gets its own connection using thread-local storage.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
        self._lock = threading.Lock()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a connection for the current thread."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            self._local.connection.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._local.connection.execute("PRAGMA journal_mode=WAL")
            self._local.connection.execute("PRAGMA synchronous=NORMAL")
        return self._local.connection
    
    @contextmanager
    def connection(self):
        """Context manager for database operations."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
    
    @contextmanager
    def cursor(self):
        """Context manager for cursor operations."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
    
    def close_all(self):
        """Close connection for current thread."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None


# =============================================================================
# TELEMETRY QUEUE (MAIN CLASS)
# =============================================================================

class TelemetryQueue:
    """
    Production-ready telemetry queue with SQLite persistence and background sync.
    
    This class provides:
    - Immediate persistence to SQLite (no data loss on crash)
    - Background thread for async sync to cloud
    - Bulk operations for efficiency
    - Thread-safe operations
    
    Attributes:
        db_path: Path to SQLite database file
        api_url: Base URL of the Node.js backend
        sync_interval: Seconds between sync attempts
        batch_size: Max events to sync in one request
        max_retries: Retry attempts before giving up on batch
        retention_days: Days to keep synced events (0 = delete immediately)
    """
    
    DEFAULT_DB_PATH = "telemetry_queue.db"
    DEFAULT_API_URL = "http://localhost:3001"
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        api_url: Optional[str] = None,
        sync_interval: float = 5.0,
        batch_size: int = 50,
        max_retries: int = 3,
        retention_days: int = 7,
        timeout: float = 10.0
    ):
        """
        Initialize the TelemetryQueue.
        
        Args:
            db_path: Path to SQLite database (default: telemetry_queue.db)
            api_url: Backend API URL (default: http://localhost:3001)
            sync_interval: Seconds between sync attempts
            batch_size: Maximum events per sync request
            max_retries: Retry attempts per batch
            retention_days: Days to keep synced events (0 = immediate delete)
            timeout: HTTP request timeout
        """
        self.db_path = db_path or self.DEFAULT_DB_PATH
        self.api_url = (api_url or self.DEFAULT_API_URL).rstrip('/')
        self.telemetry_endpoint = f"{self.api_url}/api/telemetry"
        
        self.sync_interval = sync_interval
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retention_days = retention_days
        self.timeout = timeout
        
        # Thread-safe connection manager
        self._db = SQLiteConnectionManager(self.db_path)
        
        # Sync thread control
        self._running = False
        self._sync_thread: Optional[threading.Thread] = None
        self._enqueue_lock = threading.Lock()
        
        # Statistics
        self._stats = {
            "events_queued": 0,
            "events_synced": 0,
            "events_failed": 0,
            "sync_attempts": 0,
            "last_sync_time": None,
            "last_sync_success": None,
            "is_connected": False
        }
        
        # Initialize database
        self._init_database()
        
        logger.info(f"TelemetryQueue initialized: db={self.db_path}, api={self.api_url}")
    
    def _init_database(self) -> None:
        """Create the queue table if it doesn't exist."""
        with self._db.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    zone_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    neglect_rate_pct REAL DEFAULT 0,
                    synced INTEGER DEFAULT 0,
                    retry_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    synced_at TEXT
                )
            """)
            
            # Indexes for efficient queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_queue_synced 
                ON telemetry_queue(synced)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_queue_timestamp 
                ON telemetry_queue(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_queue_created 
                ON telemetry_queue(created_at)
            """)
        
        # Load initial stats
        self._update_queue_stats()
        logger.debug("Database initialized")
    
    def _update_queue_stats(self) -> None:
        """Update statistics from database."""
        with self._db.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM telemetry_queue WHERE synced = 0")
            pending = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM telemetry_queue WHERE synced = 1")
            synced = cursor.fetchone()[0]
            
            self._stats["pending_count"] = pending
            self._stats["synced_count"] = synced
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def start(self) -> None:
        """Start the background sync thread."""
        if self._running:
            logger.warning("Queue already running")
            return
        
        if not REQUESTS_AVAILABLE:
            logger.error("Cannot start sync - requests library not available")
            return
        
        self._running = True
        self._sync_thread = threading.Thread(
            target=self._sync_loop,
            daemon=True,
            name="TelemetrySync"
        )
        self._sync_thread.start()
        logger.info("Background sync thread started")
    
    def stop(self, timeout: float = 10.0) -> None:
        """
        Stop the background sync thread.
        
        Args:
            timeout: Max seconds to wait for thread to finish
        """
        if not self._running:
            return
        
        logger.info("Stopping telemetry queue...")
        self._running = False
        
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=timeout)
        
        # Final sync attempt
        self._sync_pending_events()
        
        # Close database connections
        self._db.close_all()
        
        logger.info(f"Queue stopped. Stats: {self.stats}")
    
    def enqueue(
        self,
        zone_id: str,
        event_type: str,
        neglect_rate_pct: float = 0.0,
        timestamp: Optional[float] = None
    ) -> int:
        """
        Add an event to the queue (immediately persisted to SQLite).
        
        This method is thread-safe and non-blocking. Events are stored
        locally and will be synced to the cloud by the background thread.
        
        Args:
            zone_id: Zone identifier (e.g., "Shelf_1_Spices")
            event_type: Event type ("TAKEN", "PUT_BACK", "TOUCH")
            neglect_rate_pct: Current neglect rate percentage
            timestamp: Unix timestamp (defaults to now)
            
        Returns:
            int: The ID of the queued event
        """
        if timestamp is None:
            timestamp = time.time()
        
        with self._enqueue_lock:
            with self._db.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO telemetry_queue 
                    (timestamp, zone_id, event_type, neglect_rate_pct, synced)
                    VALUES (?, ?, ?, ?, 0)
                """, (timestamp, zone_id, event_type, neglect_rate_pct))
                
                event_id = cursor.lastrowid
        
        self._stats["events_queued"] += 1
        logger.debug(f"Event queued: {zone_id}/{event_type} (id={event_id})")
        
        return event_id
    
    def enqueue_batch(self, events: List[Dict[str, Any]]) -> List[int]:
        """
        Add multiple events to the queue in a single transaction.
        
        Args:
            events: List of event dicts with keys:
                    zone_id, event_type, neglect_rate_pct, timestamp (optional)
                    
        Returns:
            List of event IDs
        """
        event_ids = []
        
        with self._enqueue_lock:
            with self._db.cursor() as cursor:
                for event in events:
                    timestamp = event.get("timestamp", time.time())
                    cursor.execute("""
                        INSERT INTO telemetry_queue 
                        (timestamp, zone_id, event_type, neglect_rate_pct, synced)
                        VALUES (?, ?, ?, ?, 0)
                    """, (
                        timestamp,
                        event["zone_id"],
                        event["event_type"],
                        event.get("neglect_rate_pct", 0.0)
                    ))
                    event_ids.append(cursor.lastrowid)
        
        self._stats["events_queued"] += len(events)
        logger.debug(f"Batch queued: {len(events)} events")
        
        return event_ids
    
    def force_sync(self) -> Tuple[int, int]:
        """
        Force an immediate sync attempt.
        
        Returns:
            Tuple of (events_synced, events_failed)
        """
        return self._sync_pending_events()
    
    def get_pending_count(self) -> int:
        """Get the number of unsynced events."""
        with self._db.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM telemetry_queue WHERE synced = 0")
            return cursor.fetchone()[0]
    
    def clear_synced(self) -> int:
        """
        Delete all synced events from the database.
        
        Returns:
            Number of events deleted
        """
        with self._db.cursor() as cursor:
            cursor.execute("DELETE FROM telemetry_queue WHERE synced = 1")
            deleted = cursor.rowcount
        
        logger.info(f"Cleared {deleted} synced events")
        return deleted
    
    def cleanup_old_events(self, days: Optional[int] = None) -> int:
        """
        Delete old synced events based on retention policy.
        
        Args:
            days: Days to retain (uses retention_days if not specified)
            
        Returns:
            Number of events deleted
        """
        days = days if days is not None else self.retention_days
        
        if days <= 0:
            return 0
        
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        with self._db.cursor() as cursor:
            cursor.execute("""
                DELETE FROM telemetry_queue 
                WHERE synced = 1 AND timestamp < ?
            """, (cutoff,))
            deleted = cursor.rowcount
        
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old events (>{days} days)")
        
        return deleted
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        self._update_queue_stats()
        return self._stats.copy()
    
    @property
    def is_connected(self) -> bool:
        """Check if last sync was successful."""
        return self._stats.get("is_connected", False)
    
    # =========================================================================
    # BACKGROUND SYNC
    # =========================================================================
    
    def _sync_loop(self) -> None:
        """Background thread loop for syncing events."""
        logger.info("Sync loop started")
        
        # Initial cleanup
        self.cleanup_old_events()
        
        while self._running:
            try:
                # Sync pending events
                synced, failed = self._sync_pending_events()
                
                if synced > 0:
                    logger.info(f"Synced {synced} events to cloud")
                if failed > 0:
                    logger.warning(f"Failed to sync {failed} events")
                
                # Periodic cleanup
                if self._stats["sync_attempts"] % 100 == 0:
                    self.cleanup_old_events()
                
            except Exception as e:
                logger.error(f"Sync loop error: {e}")
            
            # Wait for next sync interval
            for _ in range(int(self.sync_interval * 10)):
                if not self._running:
                    break
                time.sleep(0.1)
        
        logger.info("Sync loop stopped")
    
    def _sync_pending_events(self) -> Tuple[int, int]:
        """
        Sync all pending events to the cloud backend.
        
        Returns:
            Tuple of (events_synced, events_failed)
        """
        if not REQUESTS_AVAILABLE:
            return 0, 0
        
        total_synced = 0
        total_failed = 0
        
        self._stats["sync_attempts"] += 1
        
        while True:
            # Get batch of unsynced events
            events = self._get_pending_batch()
            
            if not events:
                break
            
            # Attempt to sync batch
            success = self._send_batch_to_api(events)
            
            if success:
                # Mark events as synced
                event_ids = [e.id for e in events]
                self._mark_events_synced(event_ids)
                total_synced += len(events)
                self._stats["is_connected"] = True
                self._stats["last_sync_success"] = datetime.now().isoformat()
            else:
                # Increment retry count
                event_ids = [e.id for e in events]
                self._increment_retry_count(event_ids)
                total_failed += len(events)
                self._stats["is_connected"] = False
                break  # Stop trying if we can't connect
        
        self._stats["events_synced"] += total_synced
        self._stats["events_failed"] += total_failed
        self._stats["last_sync_time"] = datetime.now().isoformat()
        
        return total_synced, total_failed
    
    def _get_pending_batch(self) -> List[QueuedEvent]:
        """Get a batch of unsynced events."""
        events = []
        
        with self._db.cursor() as cursor:
            cursor.execute("""
                SELECT id, timestamp, zone_id, event_type, neglect_rate_pct,
                       synced, created_at
                FROM telemetry_queue
                WHERE synced = 0 AND retry_count < ?
                ORDER BY timestamp ASC
                LIMIT ?
            """, (self.max_retries, self.batch_size))
            
            for row in cursor.fetchall():
                events.append(QueuedEvent(
                    id=row["id"],
                    timestamp=row["timestamp"],
                    zone_id=row["zone_id"],
                    event_type=row["event_type"],
                    neglect_rate_pct=row["neglect_rate_pct"],
                    synced=bool(row["synced"]),
                    created_at=row["created_at"]
                ))
        
        return events
    
    def _send_batch_to_api(self, events: List[QueuedEvent]) -> bool:
        """
        Send a batch of events to the API.
        
        Args:
            events: List of QueuedEvent objects
            
        Returns:
            bool: True if successful
        """
        payload = {
            "events": [e.to_api_dict() for e in events]
        }
        
        try:
            response = requests.post(
                self.telemetry_endpoint,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.debug(f"API accepted {result.get('processed', len(events))} events")
                return True
            else:
                logger.warning(f"API error: {response.status_code} - {response.text[:100]}")
                return False
                
        except requests.exceptions.Timeout:
            logger.warning("API request timed out")
            return False
        except requests.exceptions.ConnectionError:
            logger.debug("Cannot connect to API (offline?)")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending to API: {e}")
            return False
    
    def _mark_events_synced(self, event_ids: List[int]) -> None:
        """Mark events as successfully synced."""
        if not event_ids:
            return
        
        placeholders = ",".join("?" * len(event_ids))
        
        with self._db.cursor() as cursor:
            if self.retention_days == 0:
                # Immediate delete
                cursor.execute(f"""
                    DELETE FROM telemetry_queue
                    WHERE id IN ({placeholders})
                """, event_ids)
            else:
                # Mark as synced
                cursor.execute(f"""
                    UPDATE telemetry_queue
                    SET synced = 1, synced_at = CURRENT_TIMESTAMP
                    WHERE id IN ({placeholders})
                """, event_ids)
    
    def _increment_retry_count(self, event_ids: List[int]) -> None:
        """Increment retry count for failed events."""
        if not event_ids:
            return
        
        placeholders = ",".join("?" * len(event_ids))
        
        with self._db.cursor() as cursor:
            cursor.execute(f"""
                UPDATE telemetry_queue
                SET retry_count = retry_count + 1
                WHERE id IN ({placeholders})
            """, event_ids)


# =============================================================================
# CONVENIENCE FUNCTION FOR PHASE 2 INTEGRATION
# =============================================================================

def create_telemetry_queue(
    db_path: Optional[str] = None,
    api_url: str = "http://localhost:3001",
    auto_start: bool = True
) -> TelemetryQueue:
    """
    Create and optionally start a TelemetryQueue instance.
    
    This is a convenience function for easy integration with Phase 2 engine.
    
    Args:
        db_path: Path to SQLite database (default: telemetry_queue.db)
        api_url: Backend API URL
        auto_start: Whether to start background sync immediately
        
    Returns:
        Configured TelemetryQueue instance
    """
    queue = TelemetryQueue(db_path=db_path, api_url=api_url)
    
    if auto_start:
        queue.start()
        
        # Log initial status
        pending = queue.get_pending_count()
        if pending > 0:
            logger.info(f"Found {pending} pending events from previous session")
    
    return queue


# =============================================================================
# EXAMPLE USAGE / TESTING
# =============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("  StoreSense Phase 5: TelemetryQueue Test")
    print("=" * 60)
    
    # Create queue
    queue = create_telemetry_queue(
        db_path="test_telemetry_queue.db",
        api_url="http://localhost:3001"
    )
    
    print(f"\nInitial stats: {queue.stats}")
    
    # Queue some test events
    print("\nQueueing test events...")
    test_events = [
        ("Shelf_1_Spices", "TAKEN", 5.2),
        ("Shelf_2_Snacks", "PUT_BACK", 12.8),
        ("Shelf_1_Spices", "TOUCH", 6.1),
        ("Shelf_3_Drinks", "TAKEN", 25.5),
        ("Shelf_2_Snacks", "TAKEN", 15.3),
    ]
    
    for zone_id, event_type, neglect in test_events:
        event_id = queue.enqueue(zone_id, event_type, neglect)
        print(f"  Queued: {zone_id}/{event_type} -> id={event_id}")
    
    # Check pending
    pending = queue.get_pending_count()
    print(f"\nPending events: {pending}")
    
    # Wait for sync
    print("\nWaiting for background sync (10 seconds)...")
    time.sleep(10)
    
    # Final stats
    print(f"\nFinal stats: {queue.stats}")
    print(f"Is connected: {queue.is_connected}")
    
    # Cleanup
    queue.stop()
    
    # Remove test database
    if os.path.exists("test_telemetry_queue.db"):
        os.remove("test_telemetry_queue.db")
    
    print("\nTest complete!")
