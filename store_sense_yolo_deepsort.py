import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


Box = Tuple[int, int, int, int]


@dataclass
class Detection:
    bbox: Box
    confidence: float
    class_name: str


@dataclass
class ShelfROI:
    x: int
    y: int
    width: int
    height: int

    def as_box(self) -> Box:
        return (self.x, self.y, self.x + self.width, self.y + self.height)


def load_roi_from_config(config_path: str, zone_index: int = 0) -> ShelfROI:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_file.open("r", encoding="utf-8") as file:
        config = json.load(file)

    rois = config.get("rois", [])
    if not rois:
        raise ValueError(f"No ROIs found in {config_path}")
    if zone_index < 0 or zone_index >= len(rois):
        raise IndexError(f"Zone index {zone_index} out of range for {len(rois)} ROIs")

    roi = rois[zone_index]
    return ShelfROI(
        x=int(roi["x"]),
        y=int(roi["y"]),
        width=int(roi["width"]),
        height=int(roi["height"]),
    )


def fit_roi_to_frame(roi: ShelfROI, frame_width: int, frame_height: int) -> ShelfROI:
    max_x2 = roi.x + roi.width
    max_y2 = roi.y + roi.height
    if max_x2 <= frame_width and max_y2 <= frame_height:
        return roi

    scale = min(frame_width / max_x2, frame_height / max_y2)
    fitted = ShelfROI(
        x=max(0, int(round(roi.x * scale))),
        y=max(0, int(round(roi.y * scale))),
        width=max(40, int(round(roi.width * scale))),
        height=max(40, int(round(roi.height * scale))),
    )
    if fitted.x + fitted.width > frame_width:
        fitted.width = max(40, frame_width - fitted.x)
    if fitted.y + fitted.height > frame_height:
        fitted.height = max(40, frame_height - fitted.y)
    return fitted


def get_phone_camera_url(ip_address: str, port: int = 8080) -> str:
    return f"http://{ip_address}:{port}/video"


def boxes_intersect(a: Box, b: Box) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def box_center(box: Box) -> Tuple[int, int]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def draw_label(frame, text: str, origin: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    x, y = origin
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 8, y), color, -1)
    cv2.putText(frame, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)


def detect_objects(frame, model: YOLO, roi: ShelfROI, conf: float = 0.25) -> List[Detection]:
    x1, y1, x2, y2 = roi.as_box()
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return []

    results = model(crop, conf=conf, verbose=False)
    detections: List[Detection] = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0].item())
            class_name = result.names[cls_id]
            if class_name in {"person"}:
                continue

            xyxy = box.xyxy[0].cpu().numpy()
            bx1 = int(xyxy[0]) + x1
            by1 = int(xyxy[1]) + y1
            bx2 = int(xyxy[2]) + x1
            by2 = int(xyxy[3]) + y1
            detections.append(
                Detection(
                    bbox=(bx1, by1, bx2, by2),
                    confidence=float(box.conf[0].item()),
                    class_name=class_name,
                )
            )
    return detections


def detect_hands(frame, hand_tracker) -> List[Box]:
    if hand_tracker is None:
        return []

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand_tracker.process(rgb)
    if not results.multi_hand_landmarks:
        return []

    frame_h, frame_w = frame.shape[:2]
    boxes: List[Box] = []
    for hand_landmarks in results.multi_hand_landmarks:
        xs = [lm.x for lm in hand_landmarks.landmark]
        ys = [lm.y for lm in hand_landmarks.landmark]
        x1 = max(0, int(min(xs) * frame_w) - 20)
        y1 = max(0, int(min(ys) * frame_h) - 20)
        x2 = min(frame_w, int(max(xs) * frame_w) + 20)
        y2 = min(frame_h, int(max(ys) * frame_h) + 20)
        boxes.append((x1, y1, x2, y2))
    return boxes


def track_objects(frame, detections: List[Detection], tracker: DeepSort):
    tracker_inputs = []
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        tracker_inputs.append(([x1, y1, x2 - x1, y2 - y1], detection.confidence, detection.class_name))
    return tracker.update_tracks(tracker_inputs, frame=frame)


def detect_actions(
    tracks,
    roi: ShelfROI,
    state: Dict,
    hand_boxes: List[Box],
    now: float,
    disappear_after: float,
    cooldown: float,
    require_hand: bool,
) -> List[str]:
    roi_box = roi.as_box()
    hand_in_roi = any(boxes_intersect(hand_box, roi_box) for hand_box in hand_boxes)
    if hand_in_roi:
        state["last_hand_in_roi"] = now

    seen_ids = set()
    actions: List[str] = []

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        bbox = (int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3]))
        center = box_center(bbox)
        inside_roi = roi_box[0] <= center[0] <= roi_box[2] and roi_box[1] <= center[1] <= roi_box[3]
        if not inside_roi:
            continue

        seen_ids.add(track_id)
        is_new_track = track_id not in state["active_track_ids"]
        state["active_track_ids"].add(track_id)
        state["last_seen"][track_id] = now
        state["track_boxes"][track_id] = bbox

        recent_hand = (now - state["last_hand_in_roi"]) <= 1.5 if require_hand else True
        if is_new_track and recent_hand and (now - state["last_put_back_time"]) > cooldown:
            actions.append(f"PUT_BACK: track {track_id}")
            state["last_put_back_time"] = now

    stale_ids = []
    for track_id in list(state["active_track_ids"]):
        if track_id in seen_ids:
            continue

        last_seen = state["last_seen"].get(track_id, now)
        if now - last_seen < disappear_after:
            continue

        recent_hand = (now - state["last_hand_in_roi"]) <= 1.5 if require_hand else True
        if recent_hand and (now - state["last_pick_time"]) > cooldown:
            actions.append(f"PICK: track {track_id}")
            state["last_pick_time"] = now
        stale_ids.append(track_id)

    for track_id in stale_ids:
        state["active_track_ids"].discard(track_id)
        state["last_seen"].pop(track_id, None)
        state["track_boxes"].pop(track_id, None)

    return actions


def parse_args():
    parser = argparse.ArgumentParser(description="StoreSense YOLOv8 + DeepSORT tracker")
    parser.add_argument("--source", default="0", help="Video source index or path")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 weights path")
    parser.add_argument("--roi", default=None, help="Shelf ROI as x,y,w,h (overrides config ROI)")
    parser.add_argument("--config", default="config.json", help="Config file used to load ROI by default")
    parser.add_argument("--zone-index", type=int, default=0, help="ROI index to load from config")
    parser.add_argument("--phone", type=str, metavar="IP_ADDRESS", help="Use phone camera via IP Webcam")
    parser.add_argument("--phone-port", type=int, default=8080, help="IP Webcam port")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--cooldown", type=float, default=2.0, help="Cooldown between duplicate events")
    parser.add_argument("--disappear-after", type=float, default=0.8, help="Seconds before missing track becomes PICK")
    parser.add_argument("--no-hands", action="store_true", help="Disable MediaPipe hand check")
    return parser.parse_args()


def create_hand_tracker(disabled: bool):
    if disabled or not MEDIAPIPE_AVAILABLE:
        return None
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.35,
    )


def main():
    args = parse_args()
    if args.roi:
        roi_values = [int(value.strip()) for value in args.roi.split(",")]
        if len(roi_values) != 4:
            raise ValueError("ROI must be x,y,w,h")
        roi = ShelfROI(*roi_values)
    else:
        roi = load_roi_from_config(args.config, args.zone_index)

    if args.phone:
        source = get_phone_camera_url(args.phone, args.phone_port)
    else:
        source = int(args.source) if args.source.isdigit() else args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        cap.release()
        raise RuntimeError("Could not read first frame from video source")

    frame_height, frame_width = first_frame.shape[:2]
    roi = fit_roi_to_frame(roi, frame_width, frame_height)

    model = YOLO(args.model)
    tracker = DeepSort(max_age=20, n_init=2, max_cosine_distance=0.3)
    hand_tracker = create_hand_tracker(args.no_hands)

    state = {
        "last_seen": {},
        "active_track_ids": set(),
        "track_boxes": {},
        "last_pick_time": 0.0,
        "last_put_back_time": 0.0,
        "last_hand_in_roi": 0.0,
    }

    try:
        while True:
            if first_frame is not None:
                frame = first_frame
                first_frame = None
                ok = True
            else:
                ok, frame = cap.read()
            if not ok or frame is None:
                break

            hand_boxes = detect_hands(frame, hand_tracker)
            detections = detect_objects(frame, model, roi, conf=args.conf)
            tracks = track_objects(frame, detections, tracker)
            actions = detect_actions(
                tracks=tracks,
                roi=roi,
                state=state,
                hand_boxes=hand_boxes,
                now=time.time(),
                disappear_after=args.disappear_after,
                cooldown=args.cooldown,
                require_hand=not args.no_hands,
            )

            x, y, w, h = roi.x, roi.y, roi.width, roi.height
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            draw_label(frame, "Shelf ROI", (x, y), (0, 255, 255))

            for hand_box in hand_boxes:
                hx1, hy1, hx2, hy2 = hand_box
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 180, 0), 2)
                cv2.putText(frame, "HAND", (hx1, max(18, hy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2)

            for track in tracks:
                if not track.is_confirmed():
                    continue
                ltrb = track.to_ltrb()
                bbox = (int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3]))
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 220, 120), 2)
                cv2.putText(frame, f"ID {track.track_id}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 120), 2)

            banner_y = 28
            if actions:
                for action in actions:
                    print(action)
                    draw_label(frame, action, (12, banner_y), (0, 200, 255) if action.startswith("PUT_BACK") else (0, 100, 255))
                    banner_y += 30
            else:
                draw_label(frame, "Monitoring shelf interactions", (12, banner_y), (180, 180, 180))

            cv2.imshow("StoreSense YOLOv8 + DeepSORT", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if hand_tracker is not None:
            hand_tracker.close()


if __name__ == "__main__":
    main()
