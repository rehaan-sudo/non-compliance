"""
Crowd Assembly Detection — Factory Floor MVP
============================================
Detects unusual crowd gathering (clustering of people) on a factory floor
using RTMDet-Tiny person detection + DBSCAN clustering + a simple state
machine.

Single-file, no database, no Telegram — just print + log + annotated video.

Usage:
    cd person_detection_safty_v2
    python3 crowd_assembly_detection.py
"""

import os
import sys
import time
from collections import deque
from datetime import datetime, timedelta

import cv2
import numpy as np
import torch

# ── PyTorch 2.6+ compatibility ─────────────────────────────────────────────
# torch.load defaults to weights_only=True since 2.6; the official RTMDet
# checkpoint contains legacy objects that need weights_only=False.
_orig_torch_load = torch.load
torch.load = lambda *a, **kw: _orig_torch_load(
    *a, **{**kw, "weights_only": kw.get("weights_only", False)}
)

from mmdet.apis import init_detector, inference_detector
from sklearn.cluster import DBSCAN

# =============================================================================
# CONFIGURABLE CONSTANTS — tune these without digging through the code
# =============================================================================

# --- Paths ------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))

# Model files (reuse existing RTMDet-Tiny from person_Detection_v1)
MODEL_DIR = os.path.join(
    os.path.dirname(HERE), "person_Detection_v1"
)
CONFIG_PATH = os.path.join(MODEL_DIR, "rtmdet_tiny_8xb32-300e_coco.py")
CHECKPOINT_PATH = os.path.join(
    MODEL_DIR, "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
)

# Input / output
INPUT_VIDEO_PATH ="/home/ekak-11/Desktop/kp_model/non-compliance/person_detection_safty_v2/video/testing3.mp4"
OUTPUT_VIDEO_PATH = os.path.join(HERE, "output", "annotated_output.mp4")
LOG_FILE = os.path.join(HERE, "logs", "safty_events.log")

DEVICE = "cuda:0"  # "cuda:0" or "cpu"

# --- Person Detection -------------------------------------------------------
PERSON_CLASS_ID = 0       # COCO class index for "person"
CONF_THRESHOLD = 0.3      # minimum confidence to keep a detection

# --- Zone polygon (full-frame for now; change to custom polygon later) ------
# Format: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
# Full-frame polygon will be set dynamically based on video resolution at
# runtime.  Set to None to auto-detect full frame.
ZONE_POLYGON = None       # None = use entire frame

# --- Clustering -------------------------------------------------------------
CLUSTER_RADIUS_PX = 160  #120 # DBSCAN eps — max distance (pixels) between two
                           # people to be considered in the same cluster
CLUSTER_MIN_SAMPLES = 3    # DBSCAN min_samples — minimum people to form a
                           # cluster (set >= 3 so a pair doesn't trigger)

# --- Crowd thresholds -------------------------------------------------------
SHIFT_HEADCOUNT = 10       # expected number of people normally in the zone
THRESHOLD_PERCENT = 0.4    # 40% of SHIFT_HEADCOUNT = dynamic trigger threshold
                           # e.g. 10 * 0.4 = 4 people clustered → candidate

# --- State-machine timing constants -----------------------------------------
MIN_SUSTAINED_SECONDS = 4  # how long the condition must persist to CONFIRM
FLICKER_TOLERANCE_FRAMES = 2  # max consecutive frames condition may drop
                              # during CANDIDATE without resetting
COOLDOWN_SECONDS = 120   #120    # after a CONFIRMED alert, ignore triggers for
                              # this many seconds before returning to NORMAL

# --- Display / progress -----------------------------------------------------
PROGRESS_EVERY_N_FRAMES = 100  # print progress every N frames (0 = disable)

# =============================================================================
# END OF CONFIG
# =============================================================================


# ── Utility functions ──────────────────────────────────────────────────────


def point_in_polygon(px, py, polygon):
    """Return True if (px, py) is inside the polygon (list of (x,y) tuples)."""
    contour = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(px), float(py)), False) >= 0


def ts_now():
    """Current timestamp string for logging, e.g. '2026-07-21 14:32:05'."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds):
    """Return 'H:MM:SS' string for a duration in seconds."""
    return str(timedelta(seconds=int(round(seconds))))


def append_log(line):
    """Append a line to LOG_FILE. Creates logs/ dir if missing."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
        f.flush()


# ── Detection helpers ──────────────────────────────────────────────────────


def load_model(config_path, checkpoint_path, device):
    """Load RTMDet-Tiny and return the model."""
    print(f"[INFO] Loading RTMDet-Tiny on {device} ...")
    model = init_detector(config_path, checkpoint_path, device=device)
    print("[INFO] Model loaded.")
    return model


def detect_persons(model, frame, conf_threshold=CONF_THRESHOLD):
    """Run inference and return list of person bboxes [x1,y1,x2,y2].

    Returns:
        list of (x1, y1, x2, y2) int tuples for person detections.
    """
    result = inference_detector(model, frame)
    if result is None:
        return []

    bboxes = result.pred_instances.bboxes
    scores = result.pred_instances.scores
    labels = result.pred_instances.labels

    persons = []
    for bbox, score, label in zip(bboxes, scores, labels):
        if int(label) != PERSON_CLASS_ID:
            continue
        if float(score) < conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        persons.append((x1, y1, x2, y2))
    return persons


def get_centroids(bboxes, zone_polygon):
    """Compute centroids of bboxes that fall inside zone_polygon.

    Returns:
        centroids:  list of (cx, cy) float tuples
        in_zone:    bool list parallel to bboxes — True if centroid in zone
    """
    centroids = []
    in_zone = []
    for (x1, y1, x2, y2) in bboxes:
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        inside = point_in_polygon(cx, cy, zone_polygon)
        in_zone.append(inside)
        if inside:
            centroids.append((cx, cy))
    return centroids, in_zone


# ── Clustering ─────────────────────────────────────────────────────────────


def cluster_persons(centroids, eps, min_samples):
    """Run DBSCAN on centroids and return cluster info.

    Returns:
        labels:       np.array of cluster labels (same length as centroids),
                      -1 = noise
        largest_size: int, number of points in the largest cluster (0 if none)
        largest_mask: np.array of bool, True for points in the largest cluster
    """
    if len(centroids) < min_samples:
        # Not enough points to even form a cluster
        empty_labels = np.full(len(centroids), -1, dtype=int)
        return empty_labels, 0, np.zeros(len(centroids), dtype=bool)

    X = np.array(centroids)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = clustering.labels_

    # Find largest cluster (ignore noise label -1)
    unique, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(unique) == 0:
        return labels, 0, np.zeros(len(centroids), dtype=bool)

    largest_label = unique[np.argmax(counts)]
    largest_size = int(np.max(counts))
    largest_mask = labels == largest_label

    return labels, largest_size, largest_mask


# ── Drawing ────────────────────────────────────────────────────────────────


def draw_overlays(frame, bboxes, in_zone, centroids, cluster_labels,
                  largest_mask, state, frame_idx, total_frames, fps,
                  cluster_size, threshold_val):
    """Draw bounding boxes, cluster highlights, and status overlays."""
    h, w = frame.shape[:2]

    # --- Person bounding boxes ----------------------------------------------
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        color = (0, 255, 0)  # green = person
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"person", (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # --- Zone polygon -------------------------------------------------------
    if len(bboxes) > 0:
        # Draw zone as a semi-transparent overlay
        zone_np = np.array(ZONE_POLYGON, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [zone_np], isClosed=True,
                      color=(255, 255, 0), thickness=2)
        cv2.putText(frame, "ZONE", (ZONE_POLYGON[0][0], ZONE_POLYGON[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # --- Cluster highlight --------------------------------------------------
    if largest_mask.any():
        cluster_pts = np.array([centroids[i] for i in range(len(centroids))
                                if largest_mask[i]], dtype=np.int32)
        if len(cluster_pts) >= 3:
            hull = cv2.convexHull(cluster_pts.reshape(-1, 1, 2))
            cv2.polylines(frame, [hull], isClosed=True,
                          color=(0, 0, 255), thickness=3)

            # Fill with semi-transparent red
            overlay = frame.copy()
            cv2.fillPoly(overlay, [hull], (0, 0, 255))
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        elif len(cluster_pts) >= 1:
            # Draw circle around cluster centre
            centre = cluster_pts.mean(axis=0).astype(int)
            radius = int(np.max(np.linalg.norm(cluster_pts - centre, axis=1)))
            cv2.circle(frame, tuple(centre), max(radius, CLUSTER_RADIUS_PX),
                       (0, 0, 255), 3)

    # --- Draw centroid dots --------------------------------------------------
    for i, (cx, cy) in enumerate(centroids):
        dot_color = (0, 0, 255) if cluster_labels[i] != -1 else (255, 255, 255)
        cv2.circle(frame, (int(cx), int(cy)), 4, dot_color, -1)

    # --- Status overlay -----------------------------------------------------
    state_colors = {
        "NORMAL": (0, 255, 0),
        "CANDIDATE": (0, 165, 255),
        "CONFIRMED": (0, 0, 255),
        "COOLDOWN": (128, 128, 128),
    }
    sc = state_colors.get(state, (255, 255, 255))

    y0 = 30
    cv2.putText(frame, f"FPS: {fps:.1f}  frame: {frame_idx}/{total_frames}",
                (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    y0 += 30
    cv2.putText(frame, f"Persons detected: {len(bboxes)}",
                (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y0 += 30
    cv2.putText(frame, f"STATE: {state}",
                (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, sc, 3)
    y0 += 35
    cv2.putText(frame,
                f"Cluster: {cluster_size}  |  threshold: {threshold_val:.0f}",
                (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255) if cluster_size >= threshold_val else (255, 255, 255), 2)

    return frame


# ── Main pipeline ──────────────────────────────────────────────────────────


def main():
    # --- Sanity checks ------------------------------------------------------
    if not os.path.isfile(INPUT_VIDEO_PATH):
        sys.exit(f"[ERROR] Input video not found: {INPUT_VIDEO_PATH}")
    if not os.path.isfile(CONFIG_PATH):
        sys.exit(f"[ERROR] Model config not found: {CONFIG_PATH}")
    if not os.path.isfile(CHECKPOINT_PATH):
        sys.exit(f"[ERROR] Model checkpoint not found: {CHECKPOINT_PATH}")

    # Create output dirs
    os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    # --- Device -------------------------------------------------------------
    device = DEVICE
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU.")
        device = "cpu"

    # --- Load model ---------------------------------------------------------
    model = load_model(CONFIG_PATH, CHECKPOINT_PATH, device)

    # --- Open video ---------------------------------------------------------
    print(f"[INFO] Opening video: {INPUT_VIDEO_PATH}")
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {INPUT_VIDEO_PATH}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Resolution: {width}x{height}, "
          f"Frames: {total_frames}, FPS: {fps_video:.1f}")

    # --- Zone polygon (full frame or custom) --------------------------------
    global ZONE_POLYGON
    if ZONE_POLYGON is None:
        ZONE_POLYGON = [(0, 0), (width, 0), (width, height), (0, height)]

    # --- Derived constants --------------------------------------------------
    dynamic_threshold = SHIFT_HEADCOUNT * THRESHOLD_PERCENT
    sustained_frames_needed = int(MIN_SUSTAINED_SECONDS * fps_video)
    cooldown_frames = int(COOLDOWN_SECONDS * fps_video)

    print(f"[INFO] dynamic_threshold = {dynamic_threshold:.1f} people "
          f"({SHIFT_HEADCOUNT} * {THRESHOLD_PERCENT})")
    print(f"[INFO] sustained_frames_needed = {sustained_frames_needed} "
          f"({MIN_SUSTAINED_SECONDS}s @ {fps_video:.1f} fps)")
    print(f"[INFO] cooldown_frames = {cooldown_frames} "
          f"({COOLDOWN_SECONDS}s)")
    print(f"[INFO] DBSCAN: eps={CLUSTER_RADIUS_PX}px, "
          f"min_samples={CLUSTER_MIN_SAMPLES}")
    print(f"[INFO] Zone polygon: {ZONE_POLYGON}")

    # --- Video writer -------------------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps_video,
                                 (width, height))
    if not out_writer.isOpened():
        sys.exit(f"[ERROR] Cannot create output video: {OUTPUT_VIDEO_PATH}")
    print(f"[INFO] Writing annotated output to: {OUTPUT_VIDEO_PATH}")

    # --- Write log header ---------------------------------------------------
    append_log(f"# Crowd Assembly Detection Log — started at {ts_now()}")
    append_log(f"# Video: {INPUT_VIDEO_PATH}")
    append_log(f"# Threshold: {dynamic_threshold:.1f} people "
               f"(SHIFT_HEADCOUNT={SHIFT_HEADCOUNT}, "
               f"THRESHOLD_PERCENT={THRESHOLD_PERCENT})")
    append_log(f"# DBSCAN: eps={CLUSTER_RADIUS_PX}, "
               f"min_samples={CLUSTER_MIN_SAMPLES}")
    append_log("#")

    # --- State machine variables --------------------------------------------
    state = "NORMAL"
    candidate_start_time = 0.0       # timestamp (seconds from video start)
    candidate_frame_count = 0        # frames in CANDIDATE state
    flicker_count = 0                # consecutive frames below threshold
    cooldown_start_frame = 0         # frame index when COOLDOWN started
    confirmed_event_count = 0        # how many CONFIRMED alerts so far

    frame_idx = 0
    tick_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # --- Step 1: Person detection ---------------------------------------
        bboxes = detect_persons(model, frame)

        # --- Step 2-3: Centroids + zone filter ------------------------------
        centroids, in_zone = get_centroids(bboxes, ZONE_POLYGON)

        # --- Step 4: DBSCAN clustering --------------------------------------
        cluster_labels, largest_size, largest_mask = cluster_persons(
            centroids, CLUSTER_RADIUS_PX, CLUSTER_MIN_SAMPLES
        )

        # --- Step 5-6: Threshold check --------------------------------------
        is_crowd = largest_size >= dynamic_threshold

        # --- Step 7: State machine ------------------------------------------
        if state == "NORMAL":
            if is_crowd:
                state = "CANDIDATE"
                candidate_start_time = frame_idx / fps_video
                candidate_frame_count = 1
                flicker_count = 0
                print(f"\n[STATE] NORMAL -> CANDIDATE  "
                      f"(cluster={largest_size}, thr={dynamic_threshold:.1f}) "
                      f"frame={frame_idx}")

        elif state == "CANDIDATE":
            candidate_frame_count += 1
            if is_crowd:
                flicker_count = 0  # reset flicker on good frames
            else:
                flicker_count += 1

            # Check if sustained long enough
            sustained_sec = candidate_frame_count / fps_video
            if sustained_sec >= MIN_SUSTAINED_SECONDS:
                # CONFIRMED!
                state = "CONFIRMED"
                duration_str = format_duration(sustained_sec)
                alert_msg = (
                    f"ALERT: crowd gathering confirmed | "
                    f"cluster_size={largest_size} | "
                    f"duration={duration_str} | "
                    f"frame={frame_idx} | "
                    f"time={ts_now()}"
                )
                print(f"\n{'='*60}")
                print(f"  >>> {alert_msg}")
                print(f"{'='*60}\n")
                append_log(f"CONFIRMED | {alert_msg}")
                confirmed_event_count += 1

                # Immediately enter COOLDOWN
                state = "COOLDOWN"
                cooldown_start_frame = frame_idx
                print(f"[STATE] CONFIRMED -> COOLDOWN  "
                      f"(will ignore triggers for {COOLDOWN_SECONDS}s)")

            elif flicker_count > FLICKER_TOLERANCE_FRAMES:
                # Too much flicker, reset to NORMAL
                state = "NORMAL"
                flicker_count = 0
                candidate_frame_count = 0
                print(f"[STATE] CANDIDATE -> NORMAL  "
                      f"(flicker exceeded {FLICKER_TOLERANCE_FRAMES} frames) "
                      f"frame={frame_idx}")

        elif state == "CONFIRMED":
            # This state is transient — handled immediately above
            pass

        elif state == "COOLDOWN":
            frames_in_cooldown = frame_idx - cooldown_start_frame
            if frames_in_cooldown >= cooldown_frames:
                state = "NORMAL"
                print(f"[STATE] COOLDOWN -> NORMAL  "
                      f"(cooldown period ended) frame={frame_idx}")

        # --- Draw everything -------------------------------------------------
        fps_elapsed = frame_idx / (time.time() - tick_start) if frame_idx > 0 else 0

        frame = draw_overlays(
            frame, bboxes, in_zone, centroids, cluster_labels,
            largest_mask, state, frame_idx, total_frames, fps_elapsed,
            largest_size, dynamic_threshold
        )

        # --- Additional COOLDOWN overlay ------------------------------------
        if state == "COOLDOWN":
            remaining = cooldown_frames - (frame_idx - cooldown_start_frame)
            remaining_sec = remaining / fps_video if fps_video > 0 else 0
            cv2.putText(frame,
                        f"COOLDOWN — next alert in {remaining_sec:.0f}s",
                        (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

        # --- Write output frame ---------------------------------------------
        out_writer.write(frame)

        # --- Progress -------------------------------------------------------
        if PROGRESS_EVERY_N_FRAMES and frame_idx % PROGRESS_EVERY_N_FRAMES == 0:
            print(f"  frame {frame_idx:5d}/{total_frames}  "
                  f"persons={len(bboxes)}  cluster={largest_size}  "
                  f"state={state}  FPS={fps_elapsed:.1f}")

    # --- Cleanup ------------------------------------------------------------
    cap.release()
    out_writer.release()

    elapsed = time.time() - tick_start
    print(f"\n[INFO] Done. {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx / elapsed:.1f} FPS avg)")
    print(f"[INFO] Confirmed events: {confirmed_event_count}")
    print(f"[INFO] Output video: {OUTPUT_VIDEO_PATH}")
    print(f"[INFO] Log file:     {LOG_FILE}")

    append_log(f"# Run completed at {ts_now()} — "
               f"frames={frame_idx}, events={confirmed_event_count}")


if __name__ == "__main__":
    main()
