
import os
import sys
import json
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
STATE_FILE = os.path.join(HERE, "session_state.json")

VIDEO_PATH = "/home/ekak/kp_test/rtmdet_roi_person_detection/video/combined_pipeline_20260629_120557.mp4"
OUTPUT_VIDEO_PATH = "/home/ekak/kp_test/rtmdet_roi_person_detection/output/annotated_output.mp4"
DEVICE = "cuda:0"          # "cuda:0" or "cpu"

# --- Sequential-video handling ---------------------------------------------
IS_FINAL_VIDEO = True

# --- Detection -----------------------------------------------------------
PERSON_CLASS_ID = 0        # COCO class index for "person"
CONF_THRESHOLD = 0.3       # minimum confidence to keep a detection

# --- ROI polygon (pixel coordinates on the original video frame) --------
ROI_POINTS = [
    (585, 396),  # Top-left
    (674, 396),  # Top-right
    (674, 454),  # Bottom-right
    (585, 454),  # Bottom-left
]

# --- ROI entry criteria (table/work-zone -> area-overlap based) -----------
ROI_COVERAGE_THRESHOLD = 0.3      # full-entry: ROI ka kam-se-kam 30% area covered ho
PARTIAL_COVERAGE_THRESHOLD = 0.1  # partial/alert: 10%-30% ke beech coverage

# --- Occupancy-session settings ------------------------------------------
EMPTY_GRACE_FRAMES = 30        # consecutive EMPTY frames chahiye pehle EXIT confirm ho
ENTRY_GRACE_FRAMES = 5         # consecutive OCCUPIED frames chahiye pehle ENTRY confirm ho

SESSION_MERGE_WINDOW_SECONDS = 2

# --- Debug snapshots -------------------------------------------------------
DEBUG_SAVE_FRAMES = True      # save a .jpg on every ROI ENTRY / EXIT event

# --- Logging verbosity -----------------------------------------------------
VERBOSE_DIAG = False            # True = print [DIAG], [MERGE], [ROI count change] lines
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
    """Fraction (0-1) of the ROI polygon's area that is covered by the
    person's bounding box.

    coverage = intersection_area / roi_area
    """
    x1, y1, x2, y2 = bbox

    xs = [p[0] for p in roi_polygon]
    ys = [p[1] for p in roi_polygon]
    rx1, ry1 = min(xs), min(ys)
    rx2, ry2 = max(xs), max(ys)

    roi_area = (rx2 - rx1) * (ry2 - ry1)
    if roi_area <= 0:
        return 0.0

    ix1 = max(x1, rx1)
    iy1 = max(y1, ry1)
    ix2 = min(x2, rx2)
    iy2 = min(y2, ry2)
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0

    inter_area = (ix2 - ix1) * (iy2 - iy1)
    return inter_area / roi_area


def format_duration(seconds):
    """Return H:MM:SS string for a duration in seconds."""
    return str(timedelta(seconds=int(round(seconds))))


def video_timestamp_sec(cap):
    """Current video position in seconds (LOCAL to this video file).
    Falls back to monotonic clock."""
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


def save_debug_frame(frame, event_label, video_ts_global):
    """Save *frame* as JPEG into DEBUG_DIR with a readable timestamp name."""
    os.makedirs(DEBUG_DIR, exist_ok=True)
    ts_str = ts_to_str(video_ts_global).replace(":", "-")
    fname = os.path.join(DEBUG_DIR, f"{event_label}_{ts_str}.jpg")
    cv2.imwrite(fname, frame)


# --- Cross-video state persistence -----------------------------------------

DEFAULT_STATE = {
    "session_active": False,
    "session_start_ts": 0.0,       # global timeline (seconds), continuous across videos
    "session_last_count": 0,
    "empty_frame_count": 0,
    "pending_entry_count": 0,
    "last_exit_global_ts": None,   # for time-based merge-window check
    "cumulative_offset": 0.0,      # total seconds of footage processed in PREVIOUS runs
}


def load_state():
    """Load session state from STATE_FILE, or return fresh defaults if this
    is the first video in the sequence (no state file yet)."""
    if not os.path.isfile(STATE_FILE):
        print("[INFO] No previous session_state.json found — starting fresh "
              "(this looks like the first video in the sequence).")
        return dict(DEFAULT_STATE)
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
        # merge with defaults in case new fields were added later
        merged = dict(DEFAULT_STATE)
        merged.update(state)
        print(f"[INFO] Loaded session_state.json — "
              f"session_active={merged['session_active']}, "
              f"cumulative_offset={merged['cumulative_offset']:.1f}s")
        return merged
    except (json.JSONDecodeError, OSError) as e:
        print(f"[WARN] Could not read session_state.json ({e}); starting fresh.")
        return dict(DEFAULT_STATE)


def save_state(state):
    """Persist session state to STATE_FILE for the next video run."""
    tmp_path = STATE_FILE + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, STATE_FILE)   # atomic write, avoids corruption if interrupted


def draw_predictions(frame, preds, conf_thr, roi_polygon):
    """Draw bounding boxes for *person* detections and count how many are
    "in ROI" for occupancy purposes (table/work-zone: area-overlap based).
    """
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

        coverage = roi_coverage_ratio((x1, y1, x2, y2), roi_polygon)

        in_roi = coverage >= ROI_COVERAGE_THRESHOLD
        half_body = (not in_roi) and (coverage >= PARTIAL_COVERAGE_THRESHOLD)

        if VERBOSE_DIAG:
            print(f"  [DIAG] bbox=[{x1},{y1},{x2},{y2}] coverage={coverage:.2f} -> "
                  f"{'IN_ROI' if in_roi else ('PARTIAL' if half_body else 'OUT')}")

        if in_roi:
            color = (255, 0, 0)      # blue = full entry
        elif half_body:
            color = (0, 165, 255)    # orange = partial / alert
        else:
            color = (0, 255, 0)      # green = outside

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label_text = f"person {score:.2f}"
        if in_roi:
            label_text += " [ROI]"
            roi_count += 1
        elif half_body:
            label_text += " [PARTIAL]"

        cv2.putText(
            frame, label_text, (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
        )

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
          f"partial_thr={PARTIAL_COVERAGE_THRESHOLD}  "
          f"empty_grace={EMPTY_GRACE_FRAMES}  entry_grace={ENTRY_GRACE_FRAMES}  "
          f"merge_window={SESSION_MERGE_WINDOW_SECONDS}s  is_final_video={IS_FINAL_VIDEO}")

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

    # --- occupancy-session state (loaded from previous video, if any) -----
    state = load_state()
    session_active = state["session_active"]
    session_start_ts = state["session_start_ts"]
    session_last_count = state["session_last_count"]
    empty_frame_count = state["empty_frame_count"]
    pending_entry_count = state["pending_entry_count"]
    last_exit_global_ts = state["last_exit_global_ts"]
    cumulative_offset = state["cumulative_offset"]

    if session_active and VERBOSE_DIAG:
        print(f"  [CARRY-OVER] resuming active session from "
              f"{ts_to_str(session_start_ts)} (started in a previous video)")

    last_frame = None
    video_ts_local = 0.0
    video_ts = cumulative_offset  # global timeline value, updated every frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        video_ts_local = video_timestamp_sec(cap)
        video_ts = cumulative_offset + video_ts_local   # GLOBAL timestamp, continuous across videos

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
                pending_entry_count += 1

                # Case A: recent EXIT hua tha (isi video mein ya PICHLI video
                # ke end mein) aur merge-window (seconds) ke andar re-entry ho
                # rahi hai -> same session resume, koi naya ENTRY log nahi.
                if (last_exit_global_ts is not None and
                        (video_ts - last_exit_global_ts) <= SESSION_MERGE_WINDOW_SECONDS):
                    session_active = True
                    session_last_count = roi_count
                    pending_entry_count = 0
                    last_exit_global_ts = None
                    if VERBOSE_DIAG:
                        print(f"  [MERGE] re-entry within {SESSION_MERGE_WINDOW_SECONDS}s "
                              f"window, resuming session from {ts_to_str(session_start_ts)}")

                # Case B: fresh entry -- confirm only after ENTRY_GRACE_FRAMES
                # consecutive occupied frames (filters single-frame flicker).
                elif pending_entry_count >= ENTRY_GRACE_FRAMES:
                    session_active = True
                    session_start_ts = video_ts
                    session_last_count = roi_count
                    pending_entry_count = 0
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
            pending_entry_count = 0  # flicker over, reset the entry-confirmation counter

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

                    last_exit_global_ts = video_ts   # track for time-based merge-window check

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
        elif pending_entry_count > 0:
            cv2.putText(frame, f"Session: CONFIRMING ({pending_entry_count}/{ENTRY_GRACE_FRAMES})",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

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

    # --- end-of-video handling ---------------------------------------------
    # video ki total duration nikalo taaki agli video ke liye cumulative
    # offset aage badha sakein (global timeline continuous rakhne ke liye)
    video_duration = video_ts_local if video_ts_local > 0 else (
        total_frames / fps_video if fps_video > 0 else 0.0
    )
    new_cumulative_offset = cumulative_offset + video_duration

    if session_active:
        if IS_FINAL_VIDEO:
            # Ye sequence ki AAKHRI video hai -- ab genuinely pata nahi ki
            # person ROI se bahar gaya ya nahi, footage yahi khatam ho gayi.
            # Isliye REAL exit nahi maana jaayega, alag tag ke saath likha
            # jaayega taaki analytics confuse na ho.
            duration = video_ts - session_start_ts
            incomplete_line = (f"SESSION INCOMPLETE | start_time={ts_to_str(session_start_ts)} "
                                f"| video_end_time={ts_to_str(video_ts)} "
                                f"| duration_so_far={format_duration(duration)} "
                                f"| reason=video_ended")
            print(f"\n{incomplete_line}")
            append_log(incomplete_line)
            if DEBUG_SAVE_FRAMES and last_frame is not None:
                save_debug_frame(last_frame, "INCOMPLETE", video_ts)
            # final video hai, ab session ko permanently close kar do
            session_active = False
            session_start_ts = 0.0
            session_last_count = 0
            empty_frame_count = 0
        else:
            # Beech ki video hai -- session ko close NAHI karna, agli video
            # ke liye state mein carry-forward karna hai. Koi log nahi likha
            # jaayega yahan; ENTRY already pichli/isi video mein log ho chuka
            # tha, EXIT tab likha jaayega jab genuinely EMPTY_GRACE_FRAMES
            # cross hoga (chahe wo agli video ke shuruaati frames mein ho).
            print(f"\n[INFO] Session still active at end of this video "
                  f"(started {ts_to_str(session_start_ts)}) -- carrying "
                  f"forward to next video, no log written.")

    # --- persist state for the NEXT video run -------------------------------
    new_state = {
        "session_active": session_active,
        "session_start_ts": session_start_ts,
        "session_last_count": session_last_count,
        "empty_frame_count": empty_frame_count,
        "pending_entry_count": pending_entry_count,
        "last_exit_global_ts": last_exit_global_ts,
        "cumulative_offset": new_cumulative_offset,
    }
    save_state(new_state)
    print(f"[INFO] Saved session_state.json (cumulative_offset now "
          f"{new_cumulative_offset:.1f}s)")

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