"""
StoreSense Phase 4 - Test Script
=================================
Tests the backend API and sends sample telemetry data.

Usage:
    1. Start backend: cd backend && node server.js
    2. Run this script: python test_phase4.py
"""

import requests
import time
import random
import sys

API_BASE = "http://localhost:3001"

def test_health():
    """Test the health endpoint."""
    print("\n[1] Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE}/api/health", timeout=5)
        if response.status_code == 200:
            print(f"    [OK] Backend is healthy: {response.json()}")
            return True
        else:
            print(f"    [FAIL] Unexpected status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"    [FAIL] Cannot connect to {API_BASE}")
        print("    -> Make sure backend is running: cd backend && node server.js")
        return False
    except Exception as e:
        print(f"    [FAIL] Error: {e}")
        return False

def send_sample_telemetry():
    """Send sample telemetry events."""
    print("\n[2] Sending sample telemetry events...")
    
    zones = ["Shelf_1_Spices", "Shelf_2_Snacks", "Shelf_3_Beverages", "Shelf_4_Dairy"]
    events = ["TAKEN", "PUT_BACK", "TOUCH"]
    
    sent = 0
    for _ in range(20):  # Send 20 random events
        zone = random.choice(zones)
        event = random.choice(events)
        neglect_rate = random.uniform(0, 50)
        
        payload = {
            "timestamp": int(time.time()),
            "zone_id": zone,
            "event": event,
            "neglect_rate_pct": round(neglect_rate, 1)
        }
        
        try:
            response = requests.post(
                f"{API_BASE}/api/telemetry",
                json=payload,
                timeout=5
            )
            if response.status_code == 200:
                sent += 1
            else:
                print(f"    [FAIL] Failed to send: {response.status_code}")
        except Exception as e:
            print(f"    [FAIL] Error: {e}")
    
    print(f"    [OK] Sent {sent}/20 telemetry events")
    return sent > 0

def test_analytics():
    """Test the analytics summary endpoint."""
    print("\n[3] Testing analytics summary...")
    try:
        response = requests.get(f"{API_BASE}/api/analytics/summary", timeout=5)
        if response.status_code == 200:
            data = response.json()
            summary = data.get('summary', {})
            zones = data.get('zones', [])
            alerts = data.get('alerts', [])
            
            print(f"    [OK] Summary retrieved:")
            print(f"      - Total zones: {summary.get('total_zones', 0)}")
            print(f"      - Total interactions: {summary.get('total_interactions', 0)}")
            print(f"      - Items taken: {summary.get('total_taken', 0)}")
            print(f"      - Items put back: {summary.get('total_put_back', 0)}")
            
            if zones:
                print(f"\n    Zone Details:")
                for zone in zones:
                    status_icon = {"HOT": "[HOT]", "COLD": "[COLD]", "TRAFFIC_TRAP": "[TRAP]", "NORMAL": "[OK]"}.get(zone['status'], "?")
                    print(f"      - {zone['zone_id']}: {zone['total_taken']} taken, {zone['total_put_back']} put back {status_icon}")
            
            if alerts:
                print(f"\n    Alerts ({len(alerts)}):")
                for alert in alerts[:5]:  # Show max 5 alerts
                    print(f"      - [{alert['type']}] {alert['message']}")
            
            return True
        else:
            print(f"    [FAIL] Unexpected status: {response.status_code}")
            return False
    except Exception as e:
        print(f"    [FAIL] Error: {e}")
        return False

def test_recent_events():
    """Test the recent events endpoint."""
    print("\n[4] Testing recent events...")
    try:
        response = requests.get(f"{API_BASE}/api/events/recent?limit=5", timeout=5)
        if response.status_code == 200:
            data = response.json()
            events = data.get('events', [])
            print(f"    [OK] Recent events: {data.get('count', 0)} total")
            for evt in events[:3]:
                print(f"      - {evt['zone_id']}: {evt['event_type']} @ {evt.get('timestamp_formatted', evt['timestamp'])}")
            return True
        else:
            print(f"    [FAIL] Unexpected status: {response.status_code}")
            return False
    except Exception as e:
        print(f"    [FAIL] Error: {e}")
        return False

def main():
    print("=" * 60)
    print("  STORESENSE Phase 4 - API Test")
    print("=" * 60)
    
    # Test health first
    if not test_health():
        print("\n" + "=" * 60)
        print("  FAILED: Backend not running!")
        print("  Please start: cd backend && node server.js")
        print("=" * 60)
        sys.exit(1)
    
    # Send sample data
    send_sample_telemetry()
    
    # Test analytics
    test_analytics()
    
    # Test recent events
    test_recent_events()
    
    print("\n" + "=" * 60)
    print("  TEST COMPLETE")
    print("  Open http://localhost:3000 in browser for dashboard")
    print("=" * 60)

if __name__ == "__main__":
    main()
