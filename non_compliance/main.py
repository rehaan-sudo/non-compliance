"""
RTMDet-Tiny Person Detection + ROI Occupancy Session Tracking.

All settings are hardcoded in the CONFIG block below — no command-line
arguments needed. Just edit the values and run:

    python3 main.py
"""

import os
import sys
import time
from datetime import timedelta

import cv2
import numpy as np
import torch
from mmdet.apis import init_detector, inference_detector


# =============================================================================
# CONFIG — edit these values directly, no CLI arguments needed
# =============================================================================

# --- Paths -------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "rtmdet_tiny_8xb32-300e_coco.py")
CHECKPOINT_PATH = os.path.join(
    HERE, "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
)
LOG_DIR = os.path.join(HERE, "logs")
LOG_FILE = os.path.join(LOG_DIR, "session_log.txt")
DEBUG_DIR = os.path.join(HERE, "debug_frames")

VIDEO_PATH = "/home/ekak/kp_test/rtmdet_roi_person_detection/video/multi_roi_pipeline_20260714_111420.mp4"
OUTPUT_VIDEO_PATH = "/home/ekak/kp_test/rtmdet_roi_person_detection/output/annotated_output.mp4" # e.g. "output.mp4" to save annotated video; None = no video output
DEVICE = "cuda:0"          # "cuda:0" or "cpu"

# --- Detection -----------------------------------------------------------
PERSON_CLASS_ID = 0        # COCO class index for "person"
CONF_THRESHOLD = 0.3       # minimum confidence to keep a detection

# --- ROI polygon (pixel coordinates on the original video frame) --------
ROI_POINTS = [
    (554, 506),  # top-left
    (694, 506),  # top-right
    (694, 582),  # bottom-right
    (554, 582),  # bottom-left
]
# --- Occupancy-session settings ------------------------------------------
ROI_COVERAGE_THRESHOLD = 0.5   # intersection / bbox_area must be >= this
EMPTY_GRACE_FRAMES = 30        # consecutive empty frames before closing a session
EMPTY_GRACE_FRAMES = 30          # already hai
ENTRY_GRACE_FRAMES = 5           # naya: ye kai consecutive frames chahiye ROI-occupied se pehle ENTRY confirm ho
SESSION_MERGE_WINDOW = 45  
# --- Debug snapshots -------------------------------------------------------
DEBUG_SAVE_FRAMES = True      # save a .jpg on every ROI ENTRY / EXIT event

# --- Logging verbosity -----------------------------------------------------
VERBOSE_DIAG = False            # True = print [DIAG] and [ROI count change] lines
PROGRESS_EVERY_N_FRAMES = 100   # 0 disables periodic progress prints

# =============================================================================
# End of CONFIG
# =============================================================================


def has_display():
    """Return True if we can open a GUI window."""
    return os.environ.get("DISPLAY", "") != ""


def point_in_polygon(px, py, polygon):
    """Return True if (px, py) lies inside *polygon* (list of (x,y) tuples)."""
    contour = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(px), float(py)), False) >= 0


def roi_coverage_ratio(bbox, roi_polygon):
    """Fraction (0-1) of the person's bounding-box area that overlaps with
    the ROI polygon's bounding rectangle.

    coverage = intersection_area / bbox_area

    - If >= 50% of the person's box is inside the ROI → coverage >= 0.5
    - If the box is mostly outside → coverage < 0.5
    """
    x1, y1, x2, y2 = bbox
    bbox_area = (x2 - x1) * (y2 - y1)
    if bbox_area <= 0:
        return 0.0

    xs = [p[0] for p in roi_polygon]
    ys = [p[1] for p in roi_polygon]
    rx1, ry1 = min(xs), min(ys)
    rx2, ry2 = max(xs), max(ys)

    ix1 = max(x1, rx1)
    iy1 = max(y1, ry1)
    ix2 = min(x2, rx2)
    iy2 = min(y2, ry2)
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0

    inter_area = (ix2 - ix1) * (iy2 - iy1)
    return inter_area / bbox_area


def format_duration(seconds):
    """Return H:MM:SS string for a duration in seconds."""
    return str(timedelta(seconds=int(round(seconds))))


def video_timestamp_sec(cap):
    """Current video position in seconds. Falls back to monotonic clock."""
    ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    if ms and ms > 0:
        return ms / 1000.0
    return time.monotonic()


def ts_to_str(seconds):
    """HH:MM:SS.sss string from a seconds-since-start value."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def append_log(line):
    """Append *line* to LOG_FILE (creates logs/ dir if missing)."""
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
        f.flush()


def save_debug_frame(frame, event_label, video_ts):
    """Save *frame* as JPEG into DEBUG_DIR with a readable timestamp name."""
    os.makedirs(DEBUG_DIR, exist_ok=True)
    ts_str = ts_to_str(video_ts).replace(":", "-")
    fname = os.path.join(DEBUG_DIR, f"{event_label}_{ts_str}.jpg")
    cv2.imwrite(fname, frame)

def draw_predictions(frame, preds, conf_thr, roi_polygon):
    roi_count = 0
    if preds is None:
        return roi_count

    bboxes = preds.pred_instances.bboxes
    scores = preds.pred_instances.scores
    labels = preds.pred_instances.labels

    for bbox, score, label in zip(bboxes, scores, labels):
        if int(label) != PERSON_CLASS_ID:
            continue
        if float(score) < conf_thr:
            continue

        x1, y1, x2, y2 = map(int, bbox)

        # feet point = bottom-center of bbox
        foot_x = (x1 + x2) / 2.0
        foot_y = float(y2)

        # PRIMARY check: feet inside ROI polygon (fixes tall-bbox-vs-short-ROI issue)
        foot_inside = point_in_polygon(foot_x, foot_y, roi_polygon)

        # Secondary: how much of bbox overlaps ROI (for your "half body" alert case)
        coverage = roi_coverage_ratio((x1, y1, x2, y2), roi_polygon)

        in_roi = foot_inside          # full entry = feet inside zone
        half_body = (not foot_inside) and (coverage >= 0.15)  # partial overlap, feet outside

        if VERBOSE_DIAG:
            print(f"  [DIAG] bbox=[{x1},{y1},{x2},{y2}] foot=({foot_x:.0f},{foot_y:.0f}) "
                  f"foot_inside={foot_inside} coverage={coverage:.2f}")

        if in_roi:
            color = (255, 0, 0)   # blue = full entry
        elif half_body:
            color = (0, 165, 255) # orange = partial/alert
        else:
            color = (0, 255, 0)   # green = outside

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (int(foot_x), int(foot_y)), 4, color, -1)  # visualize feet point

        label_text = f"person {score:.2f}"
        if in_roi:
            label_text += " [ROI]"
            roi_count += 1
        elif half_body:
            label_text += " [PARTIAL]"

        cv2.putText(frame, label_text, (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return roi_count


def main():
    # --- sanity checks ----------------------------------------------------
    if not os.path.isfile(VIDEO_PATH):
        sys.exit(f"Video not found: {VIDEO_PATH}")
    if not os.path.isfile(CONFIG_PATH):
        sys.exit(f"Config not found: {CONFIG_PATH}")
    if not os.path.isfile(CHECKPOINT_PATH):
        sys.exit(f"Checkpoint not found: {CHECKPOINT_PATH}")

    device = DEVICE
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU.")
        device = "cpu"

    # --- load model -------------------------------------------------------
    print(f"[INFO] Loading RTMDet-Tiny on {device} ...")
    model = init_detector(CONFIG_PATH, CHECKPOINT_PATH, device=device)
    print("[INFO] Model loaded.")

    # --- open video -------------------------------------------------------
    print(f"[INFO] Using video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {VIDEO_PATH}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Resolution: {width}x{height}, Frames: {total_frames}, Native FPS: {fps_video:.1f}")
    print(f"[INFO] ROI_POINTS: {ROI_POINTS}")
    print(f"[INFO] conf={CONF_THRESHOLD}  coverage_thr={ROI_COVERAGE_THRESHOLD}  "
          f"grace_frames={EMPTY_GRACE_FRAMES}")

    # --- determine output mode --------------------------------------------
    use_display = OUTPUT_VIDEO_PATH is None and has_display()
    out_writer = None

    if OUTPUT_VIDEO_PATH:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps_video, (width, height))
        if not out_writer.isOpened():
            sys.exit(f"Cannot create output video: {OUTPUT_VIDEO_PATH}")
        print(f"[INFO] Writing output to: {OUTPUT_VIDEO_PATH}")
    elif use_display:
        window_name = "RTMDet-Tiny Person Detection (q=quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        print("[INFO] Display mode - press 'q' to quit.")
    else:
        print("[INFO] Headless mode (no OUTPUT_VIDEO_PATH, no DISPLAY). "
              "Running inference only.")

    frame_idx = 0
    tick_start = time.time()

    # --- occupancy-session state -----------------------------------------
    session_active = False
    session_start_ts = 0.0
    session_last_count = 0
    empty_frame_count = 0
    last_frame = None
    video_ts = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        video_ts = video_timestamp_sec(cap)

        # inference
        result = inference_detector(model, frame)

        # draw ROI polygon
        roi_np = np.array(ROI_POINTS, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [roi_np], isClosed=True, color=(255, 255, 0), thickness=2)

        # ROI label
        label_x, label_y = ROI_POINTS[0]
        cv2.putText(frame, "ROI", (label_x, label_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(frame, "ROI", (label_x, label_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # draw bounding boxes + count ROI persons
        roi_count = draw_predictions(frame, result, CONF_THRESHOLD, ROI_POINTS)

        # --------------- occupancy-session logic ------------------------
        if roi_count > 0:
            empty_frame_count = 0

            if not session_active:
                session_active = True
                session_start_ts = video_ts
                session_last_count = roi_count
                entry_line = (f"ROI ENTRY | start_time={ts_to_str(video_ts)} "
                              f"| persons={roi_count}")
                print(f"\n{entry_line}")
                append_log(entry_line)
                if DEBUG_SAVE_FRAMES:
                    save_debug_frame(frame, "ENTRY", video_ts)
            else:
                if roi_count != session_last_count:
                    if VERBOSE_DIAG:
                        print(f"  [ROI count change] {session_last_count} -> {roi_count}  "
                              f"at {ts_to_str(video_ts)}")
                    session_last_count = roi_count

        else:  # roi_count == 0
            if session_active:
                empty_frame_count += 1
                if empty_frame_count >= EMPTY_GRACE_FRAMES:
                    duration = video_ts - session_start_ts
                    exit_line = (f"ROI EXIT  | start_time={ts_to_str(session_start_ts)} "
                                 f"| exit_time={ts_to_str(video_ts)} "
                                 f"| duration={format_duration(duration)}")
                    print(f"\n{exit_line}")
                    append_log(exit_line)
                    if DEBUG_SAVE_FRAMES:
                        save_debug_frame(frame, "EXIT", video_ts)
                    session_active = False
                    session_start_ts = 0.0
                    session_last_count = 0
                    empty_frame_count = 0

        # fps overlay
        elapsed = time.time() - tick_start
        fps = frame_idx / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}  frame: {frame_idx}/{total_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ROI status overlay
        occupied = roi_count > 0
        status_color = (0, 0, 255) if occupied else (0, 255, 0)
        cv2.putText(frame, f"ROI Occupied: {'YES' if occupied else 'NO'}  |  "
                            f"Persons in ROI: {roi_count}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # session status overlay
        if session_active:
            session_dur = video_ts - session_start_ts
            cv2.putText(frame, f"Session: ACTIVE  |  duration: {format_duration(session_dur)}",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # periodic progress
        if PROGRESS_EVERY_N_FRAMES and (frame_idx % PROGRESS_EVERY_N_FRAMES == 0 or frame_idx == 1):
            print(f"  frame {frame_idx:5d}/{total_frames}  FPS: {fps:.1f}")

        # output
        if out_writer:
            out_writer.write(frame)
        elif use_display:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        last_frame = frame

    # --- end-of-video: close any still-active session ------------------
    if session_active:
        duration = video_ts - session_start_ts
        exit_line = (f"ROI EXIT  | start_time={ts_to_str(session_start_ts)} "
                     f"| exit_time={ts_to_str(video_ts)} "
                     f"| duration={format_duration(duration)} "
                     f"| [closed: end of video]")
        print(f"\n{exit_line}")
        append_log(exit_line)
        if DEBUG_SAVE_FRAMES and last_frame is not None:
            save_debug_frame(last_frame, "EXIT", video_ts)

    # --- cleanup ----------------------------------------------------------
    cap.release()
    if out_writer:
        out_writer.release()
    if use_display:
        cv2.destroyAllWindows()

    elapsed = time.time() - tick_start
    avg_fps = frame_idx / elapsed if elapsed > 0 else 0
    print(f"[INFO] Done. {frame_idx} frames in {elapsed:.1f}s ({avg_fps:.1f} FPS avg)")


if __name__ == "__main__":
    main()