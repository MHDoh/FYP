# processors.py 

import cv2, numpy as np, torch, time, math
from collections import defaultdict
from typing import Dict, Tuple, Any
import torchvision.transforms as T
from ultralytics import YOLO
from PIL import Image 

def entrance_processor(frame, helpers, channel_id, toggles):
    """
    Track people through two turnstiles, detect lab-coat compliance
    and “no-red-light” illegal entries.

    helpers  must contain (pre-built one time in your main code):

        helpers = {
            # models  --------------------------------------------------
            "human_model"   : YOLO("yolo11n.pt").to(device),
            "labcoat_model" : YOLO("labcoats.pt").to(device),

            # static ROIs ---------------------------------------------
            "turnstile_rois": "List of ROI tuples for turnstiles, e.g., [(x1,y1,x2,y2), ...]",
            "light_rois"    : "List of ROI tuples for lights, e.g., [(x1,y1,x2,y2), ...]",

            # Optional processing parameters --------------------------
            "entrance_containment_threshold": 0.30,
            "entrance_color_dominance_ratio": 0.30,
            "entrance_font_scale": 0.6,
            "entrance_font_thick": 2,

            # persistent state  (filled automatically on first call) --
            # "entrance_state": {...}  ← we create this if missing
        }

    Returns
    -------
    overlay : np.ndarray
        Annotated copy of the input frame.
    metrics : dict
        {"people_in": int,
         "people_out": int,
         "illegal": int,
         "labcoat_entries": int}
    """

    overlay = frame.copy() # Create a copy for drawing

    # --- Essential components from helpers ---
    human_model = helpers.get("human_model")
    labcoat_model = helpers.get("labcoat_model")
    turnstile_rois = helpers.get("turnstile_rois")
    light_rois = helpers.get("light_rois")

    # --- Check for missing essential components ---
    missing_components = []
    if human_model is None: missing_components.append("human_model")
    if labcoat_model is None: missing_components.append("labcoat_model")
    if turnstile_rois is None: missing_components.append("turnstile_rois")
    if light_rois is None: missing_components.append("light_rois")

    if missing_components:
        y_offset = 30
        for i, comp in enumerate(missing_components):
            cv2.putText(overlay, f"Error: Missing '{comp}' in helpers.", (10, y_offset + i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return overlay, {} # Return early

    # --- Optional parameters from helpers (with defaults) ---
    CONTAINMENT_THRESHOLD = helpers.get("entrance_containment_threshold", 0.30)
    COLOR_DOMINANCE_RATIO = helpers.get("entrance_color_dominance_ratio", 0.30)
    FONT_SCALE = helpers.get("entrance_font_scale", 0.6)
    FONT_THICK = helpers.get("entrance_font_thick", 2)

    # ------------------------------------------------------------
    # 0.  INITIALISE / CACHE STATE
    state = helpers.setdefault("entrance_state", {
        "active_tracks"   : {},
        "track_id_counter": 0,
        "anomaly_active"  : [False, False],
        "anomaly_waiting" : [False, False],
        "last_directions" : ["", ""],
        "people_in"       : 0,
        "people_out"      : 0,
        "illegal_entry"   : 0,
        "labcoat_entries" : 0,
    })

    device = next(human_model.model.parameters()).device  # cuda / cpu

    if not toggles.get("entrance", False):
        return overlay, {}

    # ------------------------------------------------------------
    # 1.  YOLO person + lab-coat inference

    human_results = human_model(frame, verbose=False)
    person_bboxes = [
        b.xyxy[0].cpu().numpy().astype(int)
        for b in human_results[0].boxes
        if int(b.cls[0]) == 0
    ]

    # keep these lists around even if their toggle is off
    labcoat_boxes = []

    if toggles.get("labcoat", False):
        labcoat_results = labcoat_model(frame, verbose=False)
        labcoat_boxes = [
            b.xyxy[0].cpu().numpy().astype(int)
            for b in labcoat_results[0].boxes
        ]
        for box in labcoat_boxes:
            cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), (255,0,0), 2)

    # ------------------------------------------------------------
    # 2.  Light colour helper
    def light_color(roi):
        x1,y1,x2,y2 = roi
        hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(hsv, (  0,100,100), ( 10,255,255))
        red2 = cv2.inRange(hsv, (160,100,100), (179,255,255))
        green= cv2.inRange(hsv, ( 40, 50, 50), ( 90,255,255))
        red_ratio   = cv2.countNonZero(red1|red2) / ((x2-x1)*(y2-y1))
        green_ratio = cv2.countNonZero(green)     / ((x2-x1)*(y2-y1))
        if red_ratio   > COLOR_DOMINANCE_RATIO and red_ratio   > green_ratio: return "red"
        if green_ratio > COLOR_DOMINANCE_RATIO and green_ratio > red_ratio  : return "green"
        return None

    light_colors = [light_color(r) for r in light_rois]

    # ------------------------------------------------------------
    def containment(boxA, boxB):
        xA,yA,xB,yB = max(boxA[0],boxB[0]), max(boxA[1],boxB[1]), \
                      min(boxA[2],boxB[2]), min(boxA[3],boxB[3])
        inter = max(0,xB-xA)*max(0,yB-yA)
        areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
        return inter/areaA if areaA>0 else 0

    # ------------------------------------------------------------
    # 3.  UPDATE / CREATE TRACKS INSIDE EACH TURNSTILE ROI
    if person_bboxes:
        updated_tracks = {}
        track_counts   = [0,0]                 # how many persons currently inside each ROI

        for bbox in person_bboxes:
            center = ((bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2)

            # draw the magenta bbox only if the “YOLO boxes” toggle is set
            if toggles.get("yolo", True):
                cv2.rectangle(overlay,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(255,0,255),2)

            matched = False
            for roi_idx, roi in enumerate(turnstile_rois):
                if containment(bbox, roi) < CONTAINMENT_THRESHOLD: continue
                track_counts[roi_idx] += 1

                # try to re-associate with existing track
                for t_id, tr in state["active_tracks"].items():
                    if np.linalg.norm(np.subtract(tr["centroid"], center)) < 50:
                        matched = True
                        tr["centroid"]      = center
                        tr["bbox"]          = bbox
                        tr["frames"]       += 1
                        tr["light_history"].append(light_colors[roi_idx])
                        tr["last_bottom"]   = bbox[3]
                        tr["roi_index"]     = roi_idx
                        updated_tracks[t_id]= tr
                        break

                if not matched:         # create new track
                    tid = state["track_id_counter"]; state["track_id_counter"] += 1
                    updated_tracks[tid] = {
                        "centroid":      center,
                        "bbox":          bbox,
                        "frames":        1,
                        "light_history": [light_colors[roi_idx]],
                        "first_bottom":  bbox[3],
                        "last_bottom":   bbox[3],
                        "roi_index":     roi_idx,
                    }
                break  # only one ROI can match

    # ------------------------------------------------------------
    # 4.  HANDLE TRACKS THAT HAVE LEFT THE ROI  («finalise»)
    if person_bboxes:
        for tid in set(state["active_tracks"]) - set(updated_tracks):
            tr   = state["active_tracks"][tid]
            idx  = tr["roi_index"]
            y0,y1 = turnstile_rois[idx][1], turnstile_rois[idx][3]
            from_inside  =  y0 < tr["first_bottom"] < y1
            exited_below =  tr["last_bottom"] > y1
            from_below   =  tr["first_bottom"] > y1
            exited_above =  tr["last_bottom"] <= y1

            # mark anomaly candidates
            if "red" not in tr["light_history"]:
                state["anomaly_active"][idx]  = True
                state["anomaly_waiting"][idx] = True

            direction = None
            if from_inside and exited_below:      direction = "Entering"
            elif from_below and exited_above:     direction = "Exiting"

            if direction == "Entering":
                state["people_in"] += 1

                # ---------- lab-coat compliance check ----------
                human_box        = tr["bbox"]
                wears_labcoat    = False
                best_overlap     = 0
                best_lab_box     = None
                for lab_box in labcoat_boxes:
                    ov = containment(lab_box, human_box)
                    if ov > best_overlap: best_overlap, best_lab_box = ov, lab_box
                if best_lab_box is not None:
                    # make sure this lab-coat bbox really belongs to this person
                    ok_owner = True
                    for other_tr in list(state["active_tracks"].values()) + \
                                list(updated_tracks.values()):
                        if other_tr is tr: continue
                        if containment(best_lab_box, other_tr["bbox"]) > best_overlap:
                            ok_owner = False; break
                    wears_labcoat = ok_owner

                if ("red" not in tr["light_history"]) or (not wears_labcoat):
                    state["illegal_entry"] += 1
                    if helpers.get("anomaly_cb"):
                        helpers["anomaly_cb"]("Entrance Anomaly")
                        # send anomaly to server if available
                        if helpers.get('vthread'):
                            helpers['vthread'].send_anomaly_to_server(
                                "Entrance Anomaly", "entrance", {'channel': idx}
                            )

            elif direction == "Exiting":
                state["people_out"] += 1

        # swap track dictionaries
        state["active_tracks"] = updated_tracks

        # clear anomaly flags when turnstile is empty again
        for idx in (0,1):
            empty = not any(tr["roi_index"]==idx for tr in state["active_tracks"].values())
            if empty and state["anomaly_waiting"][idx]:
                state["anomaly_active"][idx]  = False
                state["anomaly_waiting"][idx] = False

    # ------------------------------------------------------------
    # 5.  DRAW STATIC OVERLAYS / TEXT

    # Always draw the static overlays (ROIs and light boxes)
    for roi in turnstile_rois:
        cv2.rectangle(overlay, roi[:2], roi[2:], (255,255,0), 2)
    for idx, roi in enumerate(light_rois):
        c   = light_colors[idx]
        col = (0,255,0) if c=="green" else (0,0,255) if c=="red" else (255,255,0)
        cv2.rectangle(overlay, roi[:2], roi[2:], col, 2)

    # Persistent summary metrics (always shown)
    h, w = overlay.shape[:2]
    stats_y = h - 90
    cv2.putText(overlay, f"People In: {state['people_in']}",
                (w-220, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(overlay, f"People Out: {state['people_out']}",
                (w-220, stats_y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    cv2.putText(overlay, f"Illegal Entries: {state['illegal_entry']}",
                (w-220, stats_y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    cv2.putText(overlay, f"Lab Coat Entries: {state['labcoat_entries']}",
                (w-220, stats_y+75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    if person_bboxes:
        base_y   = overlay.shape[0] - 10
        base_xs  = [200, 10]
        for idx,c in enumerate(light_colors):
            y = base_y
            if c in ("red","green"):
                cv2.putText(overlay, f"{c.title()} Light",
                            (base_xs[idx], y), cv2.FONT_HERSHEY_SIMPLEX,
                            FONT_SCALE, (0,0,255) if c=="red" else (0,255,0), FONT_THICK)
                y -= 25
            if track_counts[idx]:
                cv2.putText(overlay, f"Human Detected: {track_counts[idx]}",
                            (base_xs[idx], y), cv2.FONT_HERSHEY_SIMPLEX,
                            FONT_SCALE, (255,0,255), FONT_THICK); y -= 25
            if state["last_directions"][idx]:
                cv2.putText(overlay, f"Direction: {state['last_directions'][idx]}",
                            (base_xs[idx], y), cv2.FONT_HERSHEY_SIMPLEX,
                            FONT_SCALE, (255,255,255), FONT_THICK); y -= 25
            if state["anomaly_active"][idx]:
                cv2.putText(overlay, "Anomaly Detected",
                            (base_xs[idx], y), cv2.FONT_HERSHEY_SIMPLEX,
                            FONT_SCALE, (0,0,255), FONT_THICK); y -= 25

    # ------------------------------------------------------------
    metrics = {
        "people_in"      : state["people_in"],
        "people_out"     : state["people_out"],
        "illegal"        : state["illegal_entry"],
        "labcoat_entries": state["labcoat_entries"],
    }

    return overlay, metrics


#######################################################################################################################################################################
#######################################################################################################################################################################
#######################################################################################################################################################################


def exit_processor(frame: np.ndarray,
                   helpers: Dict[str, Any],
                   channel_id: int,
                   toggles
                  ) -> Tuple[np.ndarray, Dict[str,int]]:
    """
    Detect illegal exits: door opened with no object carried out.
    Returns an overlay and {"illegal_exit": count}.
    """

    orig    = frame                       # 👈 inference source
    overlay = orig.copy()                 # 👈 we draw only here

    if not toggles.get("exit", False):
        return orig.copy(), {"illegal_exit": 0}

    # ---------- thresholds ----------
    TH_OBJ_HUMAN = 0.50
    TH_OBJ_DOOR  = 0.60
    TH_HUMAN_DOOR= 0.50
    CONTOUR_AREA = 500

    # ---------- transforms ----------
    _door_tf = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    def _contain(inner, outer):
        xA = max(inner[0], outer[0]);  yA = max(inner[1], outer[1])
        xB = min(inner[2], outer[2]);  yB = min(inner[3], outer[3])
        if xB<=xA or yB<=yA: return 0.0
        inter = (xB-xA)*(yB-yA)
        return inter / ((inner[2]-inner[0])*(inner[3]-inner[1]))

    # ---------- create one-time helper objs ----------
    fgbg = helpers.setdefault("exit_fgbg",
                              cv2.createBackgroundSubtractorMOG2(
                                  history=500, varThreshold=50, detectShadows=True))
    states = helpers.setdefault("exit_states", {})
    st = states.setdefault(channel_id, {
    "object_flag"       : 0,
    "prev_door_status"  : "Closed",   # ← debounced status!
    "illegal_exit"      : 0,
    "anomaly_latched"   : False,
    "door_open_time"    : None,       # for anomaly timing
    "debounce_timer"    : None,       # ⬅ NEW start-of-flip timestamp
    "raw_door_status"   : "Closed"    # ⬅ last raw classifier output
    })


    # ---------- short aliases ----------
    door_roi     = helpers["door_roi"]
    expanded_roi = helpers["expanded_roi"]
    human_yolo   = helpers["human_model"]
    door_model   = helpers["door_model"]
    device       = next(door_model.parameters()).device

    overlay = orig.copy()

    # ---------- 1) HUMAN DETECTION ----------

    """
    persons = []
    for box in human_yolo(orig, verbose=False)[0].boxes:
        if int(box.cls[0]) == 0:
            b = box.xyxy[0].cpu().numpy()
            persons.append(np.array(b, dtype=int))
            cv2.rectangle(overlay, (int(b[0]), int(b[1])),
                               (int(b[2]), int(b[3])), (128, 0, 128), 2)
    """
    persons = []
    for box in human_yolo(orig, verbose=False)[0].boxes:
        if int(box.cls[0]) != 0:         # skip non-persons
            continue
        b = box.xyxy[0].cpu().numpy().astype(int)
        persons.append(b)
        if toggles.get("yolo", True):    # draw only if asked
            cv2.rectangle(overlay, (b[0],b[1]), (b[2],b[3]),
                          (128,0,128), 2)

    # any person inside main door ROI?
    human_inside = any(_contain(p, door_roi) >= TH_HUMAN_DOOR for p in persons)

    # ---------- 2) MOTION MASK (possible objects) ----------
    gray  = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    mask  = fgbg.apply(gray)
    mask  = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)[1]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    moving = []
    ex_x1, ex_y1, ex_x2, ex_y2 = expanded_roi          # unpack once
    for c in cnts:
        if cv2.contourArea(c) < CONTOUR_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        # ①  bounding-box of contour
        m_box = np.array([x, y, x + w, y + h])

        # ②  must be **fully inside** the blue ROI
        if not (x >= ex_x1 and y >= ex_y1 and x + w <= ex_x2 and y + h <= ex_y2):
            continue

        moving.append(m_box)           # only collect – no drawing here


    valid_objs = []
    for m in moving:
        # overlap of the moving box that sits **inside** the door ROI
        if _contain(m, door_roi) >= TH_OBJ_DOOR:
            continue

        # overlap with every human; reject as soon as one contains ≥ 90 %
        inside_human = False
        for p in persons:
            if _contain(m, p) >= TH_OBJ_HUMAN:
                inside_human = True
                break
        if inside_human:
            continue

        valid_objs.append(m)


    object_detected = len(valid_objs) > 0

    


    for (x1_, y1_, x2_, y2_) in valid_objs:
        cv2.rectangle(overlay, (x1_, y1_), (x2_, y2_), (0, 255, 255), 2)

    # ---------- 3) DOOR CLASSIFICATION ----------
    # --- Check for black door crop (skip classification on dark frames) ---
    x1, y1, x2, y2 = door_roi
    door_crop_raw = orig[y1:y2, x1:x2]

    # Skip if the ROI is all black
    if np.mean(door_crop_raw) < 10:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
        h = overlay.shape[0]
        cv2.putText(overlay, "Door: Unknown", (20, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return overlay, {
            "illegal_exit": st["illegal_exit"],
            "door_status": "Unknown",
            "object_detected": 1 if object_detected else 0,
            "human_inside": 1 if human_inside else 0
        }



    x1,y1,x2,y2 = door_roi
    door_np   = cv2.cvtColor(orig[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    door_pil  = Image.fromarray(door_np)                #  ← convert first
    door_crop = _door_tf(door_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = door_model(door_crop).argmax(1).item()
    door_status_raw = "Closed" if pred == 0 else "Open"
    now = time.time()


    prev_status = st["prev_door_status"]          # ← keep last-frame value
    # ----------  Debounce: accept a flip only if stable ≥ 1 s  ----------
    MIN_STABLE = 1.0  # seconds

    if door_status_raw != st["raw_door_status"]:
        # raw output just flipped → start / reset timer
        st["raw_door_status"] = door_status_raw
        st["debounce_timer"]  = now
    elif door_status_raw != st["prev_door_status"]:
        # raw output has been different; check how long
        if (now - st["debounce_timer"]) >= MIN_STABLE:
            # flip is stable → commit it
            st["prev_door_status"] = door_status_raw
            st["debounce_timer"]   = None
    # else: raw == prev → nothing to do

    door_status = st["prev_door_status"]     # ← use this everywhere below


    
    box_col = (0,255,0) if door_status=="Open" else (0,0,255)
    cv2.rectangle(overlay,(x1,y1),(x2,y2),box_col,2)
    cv2.rectangle(overlay,(expanded_roi[0],expanded_roi[1]),
                  (expanded_roi[2],expanded_roi[3]),(255,0,0),2)
    
    # ---------- 4) ANOMALY LOGIC ----------
    

    # -------------------------------------------------------------------
    # Ignore short (<1 s) open–close cycles (spurious classifier flips)
    # -------------------------------------------------------------------
    MIN_OPEN_TIME = 1.0      # seconds

    if door_status == "Open":

        if prev_status != "Open":                 # ← door has just opened
            st["door_open_time"] = now            # start timer
            st["object_flag"]    = 0              # reset for this cycle
            st["anomaly_latched"]= False          # clear banner

        # door is *still* open – check if something is being carried out
        elif object_detected:
            st["object_flag"] = 1                 # remember: object passed

    else:  # door_status == "Closed"

        if prev_status == "Open":                 # ← door has just closed
            open_dur = now - (st["door_open_time"] or now)
            st["door_open_time"] = None           # stopwatch no longer needed

            if open_dur >= MIN_OPEN_TIME:         # ignore <1 s flickers
                if st["object_flag"] == 0:        # nothing was carried out
                    st["illegal_exit"] += 1
                    if helpers.get("anomaly_cb"):
                        helpers["anomaly_cb"]("Exit Anomaly")
                        if helpers.get("vthread"):
                            helpers["vthread"].send_anomaly_to_server(
                                "Exit Anomaly", "exit", {'channel': channel_id}
                            )
                #st["anomaly_latched"] = True       # keep banner until next open
                    st["anomaly_latched"] = True          # show banner
                else:                                     # legal exit
                    st["anomaly_latched"] = False         # no banner
    # -------------------------------------------------------------------
    st["prev_door_status"] = door_status           # update for next frame


    # ---------- 5) TEXT OVERLAYS ----------
    h = overlay.shape[0]
    cv2.putText(overlay,f"Door: {door_status}",(20,h-40),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    if object_detected:
        cv2.putText(overlay,"Object Detected",(20,h-80),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
    if human_inside:
        cv2.putText(overlay,"Human Detected",(20,h-60),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(128,0,128),2)
    cv2.putText(overlay,f"Illegal Exit Count: {st['illegal_exit']}",
                (20,h-100),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    if st["anomaly_latched"]:
        cv2.putText(overlay, "ANOMALY DETECTED!", (20, h - 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Return metrics including door status
    return overlay, {
        "illegal_exit": st["illegal_exit"],
        "door_status": door_status,  # Include door status in metrics
        "object_detected": 1 if object_detected else 0,
        "human_inside": 1 if human_inside else 0
    }

#######################################################################################################################################################################
#######################################################################################################################################################################
#######################################################################################################################################################################


# --- Constants for the new tower_processor ---
# These can be further customized by passing them in the 'helpers' dictionary
LOWER_GREEN_TOWER  = np.array([30,   0, 25]) # HSV lower bound for green (Adjusted for very dim green sensitivity)
UPPER_GREEN_TOWER  = np.array([120,  255,255]) # HSV upper bound for green (Very broad)
GREEN_RATIO_THRESH_TOWER = 0.01  # Minimum fraction of green pixels in ROI to confirm green (Very sensitive)

def adjust_gamma_tower(image, gamma=0.1):
    """Adjusts the gamma of an image."""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def iou_tower(boxA, boxB):
    """Calculates Intersection over Union (IoU) for two bounding boxes."""
    # boxA, boxB are (x1, y1, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    denominator = float(boxAArea + boxBArea - interArea) + 1e-6
    return interArea / denominator

def tower_processor(frame,
                    helpers: Dict[str, Any],
                    channel_id: int,
                    toggles
                   ) -> Tuple[Any, Dict[str, Any]]:
    """
    Detects tower lights using a YOLO model, tracks them,
    verifies green color, times green duration, and raises 'persistent green' anomalies.
    """
    overlay = frame.copy()
    if not toggles.get("light", False): # Assuming "light" toggle controls this processor
        st = helpers.get("tower_states", {}).get(channel_id, {})
        return overlay, {
            "detected_lights_count": len(st.get("tracked_boxes", {})),
            "green_lights_confirmed_count": 0,
            "green_durations_per_id": {},
            "persistent_green_anomalies_count": 0
        }

    # --- Get model and configurations from helpers ---
    model = helpers.get("light_model") # Changed "best_light" to "light_model"
    if model is None:
        cv2.putText(overlay, "Error: light_model not found in helpers", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return overlay, {"detected_lights_count": 0, "green_lights_confirmed_count": 0, "green_durations_per_id": {}, "persistent_green_anomalies_count": 0}

    DEVICE = helpers.get("device", "cpu") 
    device_config = helpers.get("device_config")
    if device_config and hasattr(device_config, 'detection_thresholds') and isinstance(device_config.detection_thresholds, dict):
        conf_threshold = device_config.detection_thresholds.get('light_confidence', 0.5) # Default if specific key missing
    else:
        conf_threshold = helpers.get("tower_conf_threshold", 0.5) # Fallback to general helper or hardcoded default
    
    iou_thresh = helpers.get("tower_iou_thresh", 0.3)
    blink_tol = helpers.get("tower_blink_tolerance_secs", 1.5) 
    persistent_green_duration_threshold = helpers.get("tower_persistent_green_secs", 30)
    gamma_value = helpers.get("tower_gamma", 0.4) # Adjusted default gamma for stronger brightening

    # --- Persistent per-channel state ---
    tower_states = helpers.setdefault("tower_states", {})
    st = tower_states.setdefault(channel_id, {
        "tracked_boxes": {},      
        "time_counts": {},        
        "off_counts": {},         
        "next_id": 0,
        "prev_time": time.time(), 
        "anomaly_triggered_for_id": {} 
    })

    now = time.time()
    dt = now - st.get("prev_time", now) # Use .get for robustness on first call if prev_time wasn't set
    st["prev_time"] = now

    # --- Frame processing ---
    processed_frame = adjust_gamma_tower(frame, gamma=gamma_value)
    
    results = model.predict(
        source=processed_frame,
        conf=conf_threshold,
        device=DEVICE,
        verbose=False
    )[0] 

    boxes_xyxy = results.boxes.xyxy.cpu().numpy().astype(int)
    classes = results.boxes.cls.cpu().numpy().astype(int)

    current_frame_tracked_ids = set()
    detected_lights_info = [] 

    for (x1, y1, x2, y2), cls_idx in zip(boxes_xyxy, classes):
        if x2 <= x1 or y2 <= y1: 
            continue
        current_box_coords = (x1, y1, x2, y2)
        
        matched_id = None
        best_iou_score = iou_thresh
        for tid, tracked_box_coords in st["tracked_boxes"].items():
            val = iou_tower(current_box_coords, tracked_box_coords)
            if val > best_iou_score:
                best_iou_score = val
                matched_id = tid
        
        if matched_id is None:
            matched_id = st["next_id"]
            st["next_id"] += 1
            st["time_counts"][matched_id] = 0.0
            st["off_counts"][matched_id] = 0.0
            st["anomaly_triggered_for_id"][matched_id] = False

        st["tracked_boxes"][matched_id] = current_box_coords
        current_frame_tracked_ids.add(matched_id)

        color_name = f"class_{cls_idx}" # Default if no names
        if hasattr(model, 'names') and model.names and int(cls_idx) < len(model.names):
             color_name = model.names[int(cls_idx)].lower()
        
        current_pct_g = 0.0 # Initialize pct_g for the current detection
        is_green_confirmed = False
        if color_name == 'green':
            roi_crop = frame[y1:y2, x1:x2] 
            h_roi, w_roi = roi_crop.shape[:2]
            if h_roi > 0 and w_roi > 0:
                hsv = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2HSV)
                mask_g = cv2.inRange(hsv, LOWER_GREEN_TOWER, UPPER_GREEN_TOWER)
                current_pct_g = cv2.countNonZero(mask_g) / (h_roi * w_roi + 1e-6) 
                is_green_confirmed = current_pct_g >= GREEN_RATIO_THRESH_TOWER
        
        detected_lights_info.append({
            'id': matched_id, 
            'box': current_box_coords, 
            'color_name': color_name, 
            'is_green_confirmed': is_green_confirmed,
            'pct_g': current_pct_g # Store pct_g, will be non-zero only if color_name was 'green' and ROI valid
        })

        # Update time counts based on green confirmation and blink tolerance
        if is_green_confirmed:
            st["time_counts"][matched_id] = st["time_counts"].get(matched_id, 0.0) + dt
            st["off_counts"][matched_id] = 0.0 
        else: 
            st["off_counts"][matched_id] = st["off_counts"].get(matched_id, 0.0) + dt
            if st["off_counts"][matched_id] <= blink_tol:
                 st["time_counts"][matched_id] = st["time_counts"].get(matched_id, 0.0) + dt 
            else:
                st["time_counts"][matched_id] = 0.0
                st["anomaly_triggered_for_id"][matched_id] = False # Reset anomaly if light is off for too long

    # --- Remove IDs that are no longer tracked ---
    ids_to_remove = set(st["tracked_boxes"].keys()) - current_frame_tracked_ids
    for tid_remove in ids_to_remove:
        if tid_remove in st["tracked_boxes"]: del st["tracked_boxes"][tid_remove]
        # Optionally reset or keep time/off counts for untracked IDs based on desired behavior
        # For this implementation, we'll reset anomaly flag and let times persist until overwritten or explicitly cleared
        if tid_remove in st["anomaly_triggered_for_id"]: st["anomaly_triggered_for_id"][tid_remove] = False


    # --- Anomaly detection and Drawing ---
    persistent_green_anomalies_count = 0
    green_lights_currently_on_count = 0

    for light_id_tracked in list(st["tracked_boxes"].keys()): # Iterate over a copy if modifying dict
        # Find the full info for this light_id from current frame's detections
        current_light_info = next((info for info in detected_lights_info if info['id'] == light_id_tracked), None)
        
        # If not found in current detections but still in tracked_boxes (e.g. disappeared this frame)
        # its green status depends on its last known state and off_counts
        is_effectively_green = st["time_counts"].get(light_id_tracked, 0.0) > 0 
        
        if is_effectively_green:
            green_lights_currently_on_count +=1
            current_green_duration = st["time_counts"][light_id_tracked]
            if current_green_duration > persistent_green_duration_threshold:
                if not st["anomaly_triggered_for_id"].get(light_id_tracked, False):
                    persistent_green_anomalies_count += 1
                    st["anomaly_triggered_for_id"][light_id_tracked] = True 
                    
                    if helpers.get("anomaly_cb"):
                        helpers["anomaly_cb"](f"Tower Anomaly: ID {light_id_tracked} Green")
                    if helpers.get('vthread'):
                        helpers['vthread'].send_anomaly_to_server(
                            "Tower Light Anomaly", "tower", 
                            {'channel': channel_id, 'light_id': light_id_tracked, 'duration_seconds': round(current_green_duration,1)}
                        )
                    # Visual cue for anomaly on overlay (can be drawn near the box if available)
                    # This part of drawing is complex if box is not in current_light_info
        
        # Drawing logic (prefer current_light_info if available for box coords)
        display_box = st["tracked_boxes"][light_id_tracked] # Default to stored box
        display_color_name = "N/A"
        if current_light_info:
            display_box = current_light_info['box']
            display_color_name = current_light_info['color_name']
        
        bx1, by1, bx2, by2 = display_box
        box_color_draw = (0,255,0) if is_effectively_green else (128,128,128) 
        if display_color_name == 'red': box_color_draw = (0,0,255)
        elif display_color_name == 'yellow' or display_color_name == 'orange': box_color_draw = (0,255,255)
        
        cv2.rectangle(overlay, (bx1,by1), (bx2,by2), box_color_draw, 2)
        label_text = f"ID:{light_id_tracked}"
        if is_effectively_green:
            label_text += f" G:{st['time_counts'].get(light_id_tracked, 0.0):.1f}s"
        else:
            label_text += f" ({display_color_name})"
        if st["anomaly_triggered_for_id"].get(light_id_tracked, False):
             label_text += " ANOMALY!"
             cv2.putText(overlay, "ANOMALY", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)


        cv2.putText(overlay, label_text, (bx1, by2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # --- Overall status text ---
    y_pos_text = overlay.shape[0]
    cv2.putText(overlay, f"Tracked Lights: {len(st['tracked_boxes'])}", (10, y_pos_text-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(overlay, f"Anomalies: {persistent_green_anomalies_count}", (10, y_pos_text-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255) if persistent_green_anomalies_count > 0 else (255,255,255), 2)

    # --- Prepare metrics and server updates ---
    # Create a list of (id, pct_g, duration) for currently tracked and effectively green lights
    active_green_light_details = []
    for lid, secs in st["time_counts"].items():
        if st["tracked_boxes"].get(lid) and secs > 0: # Effectively green and tracked
            # Find the pct_g for this light from detected_lights_info
            light_info = next((info for info in detected_lights_info if info['id'] == lid), None)
            current_pct_g = 0.0
            if light_info and light_info['is_green_confirmed']:
                current_pct_g = light_info.get('pct_g', 0.0)
            active_green_light_details.append((lid, current_pct_g, secs))

    # Sort these green lights first by 'pct_g' (descending), then by duration (descending) as a tie-breaker
    sorted_active_green_lights = sorted(
        active_green_light_details,
        key=lambda item: (item[1], item[2]), # Sort by pct_g (item[1]), then duration (item[2])
        reverse=True
    )
    
    # Select top 4 for display purposes, rounding the seconds
    top_4_green_display_timers = {
        lid: round(secs, 1) for lid, pct_g, secs in sorted_active_green_lights[:4]
    }

    metrics = {
        "detected_lights_count": len(st["tracked_boxes"]),
        "green_lights_active_count": green_lights_currently_on_count, 
        "green_durations_per_id": top_4_green_display_timers, # Timers for display (max 4)
        "persistent_green_anomalies_count": persistent_green_anomalies_count,
    }
    
    # --- Server communication for timer updates (if vthread and method exist) ---
    if helpers.get('vthread') and hasattr(helpers['vthread'], 'send_status_update_to_server'):
        status_update_payload = {
            "channel_id": channel_id,
            "type": "tower_light_timers", 
            "data": top_4_green_display_timers 
        }
        try:
            # Assuming send_status_update_to_server expects a dictionary payload
            helpers['vthread'].send_status_update_to_server(status_update_payload)
        except Exception as e:
            # Optional: log this error appropriately instead of just printing
            print(f"FYP: Error sending tower_light_timers update to server for channel {channel_id}: {e}")
    
    return overlay, metrics

#######################################################################################################################################################################
#######################################################################################################################################################################
#######################################################################################################################################################################


def labcoat_processor(frame,
                      helpers : Dict[str, Any],
                      channel_id : int,
                      toggles) -> Tuple[np.ndarray, Dict[str,int]]:
    """
    Detect lab-coat bounding boxes and count them.

    Returns
    -------
    overlay : frame with rectangles & label
    metrics : {"labcoats": <int>}
    """

    if not toggles.get("labcoat", False):
        return frame.copy(), {"labcoats": 0}
    
    # ---------- constants & per-channel state ---------- 
    miss_secs = helpers.get("labcoat_missing_secs", 3)
    lc_states = helpers.setdefault("labcoat_states", {})
    st = lc_states.setdefault(channel_id, {
        "miss_start": None,
        "anomaly"   : False,
    })

        
    lab_model      = helpers["labcoat_model"]
    target_size    = helpers.get("frame_size", (640,640))
    conf           = helpers.get("labcoat_conf", 0.6)

    human_model = helpers["human_model"]
    human_conf = helpers.get("human_conf", 0.5)
    small_frame = cv2.resize(frame, target_size)
    person_bboxes = [
        b.xyxy[0].cpu().numpy()
        for b in human_model.predict(small_frame, conf=human_conf, verbose=False)[0].boxes
        if int(b.cls[0]) == 0
    ]
    

    # -------- inference on a resized copy (faster) -------- 
    resized   = cv2.resize(frame, target_size)
    results   = lab_model.predict(resized, conf=conf, verbose=False)
    boxes     = results[0].boxes
    count     = len(boxes)


    # -------- draw on the original-resolution overlay ----- 
    overlay   = frame.copy()
    h_scale   = frame.shape[0] / target_size[1]
    w_scale   = frame.shape[1] / target_size[0]

    for b in boxes:
        x1,y1,x2,y2 = b.xyxy[0].cpu().numpy().astype(int)
        # scale back to original coords
        x1 = int(x1 * w_scale); x2 = int(x2 * w_scale)
        y1 = int(y1 * h_scale); y2 = int(y2 * h_scale)
        cv2.rectangle(overlay,(x1,y1),(x2,y2),(255,0,0),2)

    cv2.putText(overlay,f"Lab Coats: {count}",
                (10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)
    
    detected_persons = len(person_bboxes)          # <-- rely on YOLO already run
    lab_coat_count   = len(boxes)
    
    # ---------- debounce logic ---------- 
    detected_persons = len(person_bboxes)
    now = time.time()
    # “missing” = at least one person w/o coat
    missing = detected_persons > 0 and lab_coat_count < detected_persons
    if missing:
        if st["miss_start"] is None:
            st["miss_start"] = now
        elif not st["anomaly"] and (now - st["miss_start"]) >= miss_secs:
            st["anomaly"] = True
            if helpers.get("anomaly_cb"):
                helpers["anomaly_cb"]("Lab-Coat Anomaly")
                if helpers.get('vthread'):
                    helpers['vthread'].send_anomaly_to_server(
                        "Lab-Coat Anomaly", "labcoat", {'channel': channel_id}
                    )
    else:
        st["miss_start"] = None
        st["anomaly"]    = False

    return overlay, {"labcoats": count}


#######################################################################################################################################################################
#######################################################################################################################################################################
#######################################################################################################################################################################


def density_processor(frame,
                      helpers : Dict[str,Any],
                      channel_id : int,
                      toggles: Dict[str,bool]) -> Tuple[np.ndarray, Dict[str,int]]:
    """
    Draw a coloured human-density heat-map and return the largest group size.
    helpers must contain a loaded YOLO model under "human_model".
    Optional keys:
        "human_conf"   (float)  confidence threshold    default 0.5
        "frame_size"   (w,h)    inference resolution    default (640,640)
        "density_colors" list[Tuple[B,G,R], label] for legend.
    """

    if not toggles.get("density", False):
        return frame.copy(), {"crowd": 0}

    # ---------- internal util (one-liner union-find for clusters) ---------- 
    def _find(p,i):           # quick DSU helpers for clustering circles
        if p[i]!=i: p[i]=_find(p,p[i])
        return p[i]
    def _union(p,r,a,b):
        ra,rb=_find(p,a),_find(p,b)
        if ra==rb: return
        if r[ra]<r[rb]: ra,rb=rb,ra
        p[rb]=ra
        if r[ra]==r[rb]: r[ra]+=1
    # ---------------------------------------------------------------------- 

    thr_people = helpers.get("crowd_threshold", 4)
    thr_secs   = helpers.get("crowd_secs", 5)

    dens_states = helpers.setdefault("density_states", {})
    st = dens_states.setdefault(channel_id, {
        "crowd_start": None,
        "anomaly"    : False,
    })


    yolo          = helpers["human_model"]
    conf          = helpers.get("human_conf", 0.5)
    target_size   = helpers.get("frame_size", (640,640))

    # ---------- 1. run person detector ---------- 
    small         = cv2.resize(frame, target_size)
    results       = yolo.predict(small, conf=conf, verbose=False)
    persons       = []
    for box in results[0].boxes:
        if int(box.cls[0])!=0:          # cls 0 == person
            continue
        x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
        # scale back to full-res
        x1 *= frame.shape[1]/target_size[0]
        x2 *= frame.shape[1]/target_size[0]
        y1 *= frame.shape[0]/target_size[1]
        y2 *= frame.shape[0]/target_size[1]
        persons.append((int(x1),int(y1),int(x2),int(y2)))

    # no people → nothing to do
    overlay = frame.copy()
    if not persons:
        return overlay, {"crowd": 0}

    # ---------- 2. convert boxes → circles ---------- 
    circles=[]
    for (x1,y1,x2,y2) in persons:
        cx = (x1+x2)//2
        cy = (y1+y2)//2
        r  = int(math.sqrt((x2-x1)**2+(y2-y1)**2)/2*0.75)
        circles.append((cx,cy,r))

    # ---------- 3. simple clustering by circle overlap ---------- 
    n        = len(circles)
    parent   = list(range(n))
    rank     = [0]*n
    for i in range(n):
        for j in range(i+1,n):
            (x1,y1,r1) = circles[i]
            (x2,y2,r2) = circles[j]
            if math.hypot(x1-x2,y1-y2) < r1+r2:      # overlap
                _union(parent,rank,i,j)

    # group them
    groups = {}
    for i in range(n):
        root=_find(parent,i)
        groups.setdefault(root,[]).append(i)

    max_group = max(map(len, groups.values()))

    # ---- debounce logic ---- 
    now = time.time()
    crowded = max_group >= thr_people
    if crowded:
        if st["crowd_start"] is None:
            st["crowd_start"] = now
        elif not st["anomaly"] and (now - st["crowd_start"]) >= thr_secs:
            st["anomaly"] = True
            if helpers.get("anomaly_cb"):
                helpers["anomaly_cb"]("Density Anomaly")
                if helpers.get('vthread'):
                    helpers['vthread'].send_anomaly_to_server(
                        "Density Anomaly", "crowd", {'channel': channel_id}
                    )
    else:
        st["crowd_start"] = None
        st["anomaly"]     = False

    # ---------- 4. draw density “heat-map” ---------- 
    density_colors = helpers.get("density_colors",
        [((65,105,225),1),((50,205,50),2),((255,191,0),3),
         ((255,69,0),4),((220,20,60),5)]
    )
    for root,idxs in groups.items():
        col = density_colors[min(len(idxs),5)-1][0]  # pick colour
        for i in idxs:
            x,y,r = circles[i]
            cv2.circle(overlay,(x,y),r,col,2)

    cv2.putText(overlay,f"Crowd max group: {max_group}",
                (10,55),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    
    return overlay, {"crowd": max_group}






