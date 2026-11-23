import io
import os
import sqlite3
import tempfile
import time
from contextlib import closing
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="Crack & Hole Detection", page_icon="🔍", layout="wide")

MODEL_PATH = "best_test.pt"
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detections.db")


# ---------------------------------------------------------------- database
def init_db():
    with closing(sqlite3.connect(DB_PATH)) as conn:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    time TEXT,
                    type TEXT,
                    file TEXT,
                    latitude REAL,
                    longitude REAL,
                    location_source TEXT,
                    cracks_found INTEGER,
                    holes_found INTEGER
                )
                """
            )
            cols = [r[1] for r in conn.execute("PRAGMA table_info(detections)")]
            if "video_time" not in cols:  # migrate old DB
                conn.execute("ALTER TABLE detections ADD COLUMN video_time REAL")


def add_logs(rows):
    """rows: list of (kind, name, location, source, cracks, holes, video_time)"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = [
        (now, k, n, loc[0] if loc else None, loc[1] if loc else None, s, int(c), int(h), vt)
        for k, n, loc, s, c, h, vt in rows
    ]
    with closing(sqlite3.connect(DB_PATH)) as conn:
        with conn:
            conn.executemany(
                """
                INSERT INTO detections
                (time, type, file, latitude, longitude, location_source,
                 cracks_found, holes_found, video_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                data,
            )


def add_log(kind, name, location, source, cracks, holes):
    add_logs([(kind, name, location, source, cracks, holes, None)])


def load_log() -> pd.DataFrame:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        return pd.read_sql_query(
            "SELECT time, type, file, latitude, longitude, location_source, "
            "cracks_found, holes_found, video_time FROM detections ORDER BY id",
            conn,
        )


def clear_log():
    with closing(sqlite3.connect(DB_PATH)) as conn:
        with conn:
            conn.execute("DELETE FROM detections")


init_db()


# ------------------------------------------------------------------- model
@st.cache_resource
def load_model(path: str):
    from ultralytics import YOLO
    return YOLO(path)


try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load `{MODEL_PATH}`: {e}")
    st.stop()


def kind_of(cls_id: int):
    name = model.names[int(cls_id)].lower()
    if "hole" in name:
        return "hole"
    if "crack" in name:
        return "crack"
    return None


def count_types(result) -> dict:
    counts = {"crack": 0, "hole": 0}
    if result.boxes is None or len(result.boxes) == 0:
        return counts
    for c in result.boxes.cls.int().tolist():
        k = kind_of(c)
        if k:
            counts[k] += 1
    return counts


def remove_overlaps(result, contain_thr: float):
    """Drop boxes that are mostly covered by a more confident box.
    contain_thr: 0.0-1.0, share of the smaller box that must be inside the other one.
    Set to 1.0 (or above) to disable."""
    boxes = result.boxes
    if boxes is None or len(boxes) < 2 or contain_thr >= 1.0:
        return result
    xyxy = boxes.xyxy.cpu().numpy()
    conf_ = boxes.conf.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    keep = []
    for i in np.argsort(-conf_):  # most confident first
        drop = False
        for j in keep:
            ix1, iy1 = max(xyxy[i, 0], xyxy[j, 0]), max(xyxy[i, 1], xyxy[j, 1])
            ix2, iy2 = min(xyxy[i, 2], xyxy[j, 2]), min(xyxy[i, 3], xyxy[j, 3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            smaller = max(min(areas[i], areas[j]), 1e-6)
            if inter / smaller > contain_thr:
                drop = True
                break
        if not drop:
            keep.append(int(i))
    if len(keep) == len(boxes):
        return result
    return result[sorted(keep)]


# ---------------------------------------------------------------- location
def _to_deg(dms) -> float:
    d, m, s = (float(x) for x in dms)
    return d + m / 60 + s / 3600


def read_exif(img: Image.Image):
    """Return ((lat, lon) or None, capture_time or None) from EXIF."""
    exif = img.getexif()
    if not exif:
        return None, None
    taken = exif.get_ifd(0x8769).get(36867) or exif.get(306)
    gps = exif.get_ifd(0x8825)
    if not gps:
        return None, taken
    try:
        lat, lon = _to_deg(gps[2]), _to_deg(gps[4])
        if gps.get(1, "N") == "S":
            lat = -lat
        if gps.get(3, "E") == "W":
            lon = -lon
        return (lat, lon), taken
    except (KeyError, TypeError, ZeroDivisionError):
        return None, taken


def pick_location(key: str, exif_loc=None, quiet=False):
    """Return ((lat, lon) or None, source): photo EXIF if available, else the device's location."""
    if exif_loc:
        return exif_loc, "Photo EXIF"
    try:
        from streamlit_js_eval import get_geolocation
    except ImportError:
        st.error("Install `streamlit-js-eval` (pip install streamlit-js-eval) to get the device location.")
        return None, None
    loc = get_geolocation(component_key=f"{key}_geo")
    if loc and "coords" in loc:
        return (loc["coords"]["latitude"], loc["coords"]["longitude"]), "Device location"
    if not quiet:
        st.info("📍 Allow location access in your browser to attach a location...")
    return None, None


NO_LOCATION = "Not given location"


def tidy_log(df: pd.DataFrame) -> pd.DataFrame:
    """Replace empty values (None/NaN) with a dash so the log table looks clean."""
    d = df.copy()
    d["video_time"] = d["video_time"].map(lambda v: f"{v:.1f} s" if pd.notna(v) else "-")
    for c in ("latitude", "longitude"):
        d[c] = d[c].map(lambda v: f"{v:.6f}" if pd.notna(v) else NO_LOCATION)
    d["location_source"] = d["location_source"].fillna(NO_LOCATION)
    return d


@st.cache_data(show_spinner=False, ttl=86400)
def place_name(lat: float, lon: float) -> str:
    """Area or city name for a coordinate (OpenStreetMap Nominatim, English if available).
    Raises on network errors so a failed lookup is not cached. Nominatim allows ~1 request/second."""
    time.sleep(1.1)
    r = requests.get(
        "https://nominatim.openstreetmap.org/reverse",
        params={"format": "jsonv2", "lat": lat, "lon": lon, "zoom": 14,
                "addressdetails": 1, "accept-language": "en"},
        headers={"User-Agent": "crack-detection-streamlit-app"},
        timeout=10,
    )
    r.raise_for_status()
    a = r.json().get("address", {})
    # area first, then city/town/village; the street is deliberately left out
    return (a.get("neighbourhood") or a.get("suburb") or a.get("city_district")
            or a.get("quarter") or a.get("town") or a.get("city") or a.get("village") or "")


def place_label(lat, lon, fallback: str) -> str:
    try:
        return place_name(float(lat), float(lon)) or fallback
    except Exception:
        return fallback


# ----------------------------------------------------------------- ranking
def ranked_by_location(df: pd.DataFrame) -> pd.DataFrame:
    """Group by location (~11 m grid); the location with the most defects comes first.
    Detections without a location are grouped together in one row with empty lat/lon."""
    d = df.copy()
    d["lat"] = d["latitude"].round(4)
    d["lon"] = d["longitude"].round(4)
    g = d.groupby(["lat", "lon"], as_index=False, dropna=False).agg(
        total_cracks=("cracks_found", "sum"),
        total_holes=("holes_found", "sum"),
        detections=("file", "count"),
        last_detection=("time", "max"),
    )
    g["total_defects"] = g["total_cracks"] + g["total_holes"]
    g = g.sort_values("total_defects", ascending=False).reset_index(drop=True)
    g.index = g.index + 1
    g.index.name = "rank"
    return g


# ---------------------------------------------------------------- settings
imgsz = 640
stride = 3

conf = 0.40         # confidence threshold: higher = fewer, more certain boxes
iou = 0.10          # NMS overlap: lower = removes more overlapping boxes
contain_thr = 0.50  # drop a box if this share of it lies inside a more confident box (1.0 = off)

st.title("🔍 Crack & Hole Detection")
st.caption("Upload an image or a video, set where it was captured, and detect cracks and holes.")

tab_img, tab_vid = st.tabs(["📷 Image", "🎥 Video"])

# ------------------------------------------------------------------- image
with tab_img:
    mode = st.radio("Image source", ["Upload photo", "Take photo"], horizontal=True)
    if mode == "Upload photo":
        file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="img_up")
    else:
        cam_col, _ = st.columns([1, 2])  # camera takes 1/3 of the page width
        with cam_col:
            file = st.camera_input("Capture an image")

    if file:
        raw = Image.open(io.BytesIO(file.getvalue()))
        gps, taken = read_exif(raw) if mode == "Upload photo" else (None, None)
        img = ImageOps.exif_transpose(raw).convert("RGB")

        location, source = pick_location("img", gps, quiet=True)

        # Only warn when there is no location at all (neither photo GPS nor device location)
        if not location:
            st.warning(
                "No location found yet. The photo has no GPS data, so allow location access "
                "in your browser to attach one."
            )

        left, right = st.columns(2)
        left.image(img, caption="Input image", use_container_width=True)
        with right:
            if location:
                if taken:
                    st.write(f"**Captured:** {taken}")
                st.map(pd.DataFrame({"lat": [location[0]], "lon": [location[1]]}), zoom=14)

        if st.button("Run detection", type="primary", key="img_run"):
            with st.spinner("Detecting..."):
                res = model.predict(
                    img, conf=conf, iou=iou, agnostic_nms=True, imgsz=imgsz, verbose=False
                )[0]
                res = remove_overlaps(res, contain_thr)
                annotated = res.plot(line_width=2, font_size=10)[:, :, ::-1]
                counts = count_types(res)
            st.image(
                annotated,
                caption=f"{counts['crack']} crack(s), {counts['hole']} hole(s) detected",
                use_container_width=True,
            )
            add_log(
                "image",
                getattr(file, "name", "camera.jpg"),
                location,
                source,
                counts["crack"],
                counts["hole"],
            )

# ------------------------------------------------------------------- video
with tab_vid:
    st.info(
        "Each crack/hole is tracked across frames and logged separately. "
        "All of them get the device location."
    )
    vid = st.file_uploader("Choose a video", type=["mp4", "mov", "avi", "mkv"], key="vid_up")

    if vid:
        st.video(vid)

        fixed, v_source = pick_location("vid")
        locator = (lambda t, f=fixed: f) if fixed else None

        if locator is None:
            st.warning("Waiting for the device location. Allow location access in your browser.")

        if locator is not None and st.button("Run detection on video", type="primary", key="vid_run"):
            suffix = os.path.splitext(vid.name)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(vid.getvalue())
                in_path = tmp.name
            out_path = in_path.rsplit(".", 1)[0] + "_annotated.mp4"

            cap = cv2.VideoCapture(in_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            writer = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*"mp4v"), max(fps / stride, 1), (w, h)
            )

            model.predictor = None  # reset tracker state for a new video
            seen = set()  # (kind, track_id) already registered
            events = []  # one entry per unique defect
            peak = {"crack": 0, "hole": 0}
            best, idx = [], 0
            bar = st.progress(0.0, text="Analyzing video...")

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % stride == 0:
                    t_sec = idx / fps
                    res = model.track(
                        frame, persist=True, conf=conf, iou=iou,
                        agnostic_nms=True, imgsz=imgsz, verbose=False
                    )[0]
                    res = remove_overlaps(res, contain_thr)
                    counts = count_types(res)
                    for k in peak:
                        peak[k] = max(peak[k], counts[k])

                    if res.boxes is not None and res.boxes.id is not None:
                        for tid, c in zip(
                            res.boxes.id.int().tolist(), res.boxes.cls.int().tolist()
                        ):
                            k = kind_of(c)
                            if k and (k, tid) not in seen:
                                seen.add((k, tid))
                                events.append((k, t_sec, locator(t_sec)))

                    plotted = res.plot(line_width=2, font_size=10)
                    writer.write(plotted)
                    n = counts["crack"] + counts["hole"]
                    if n > 0:
                        best.append((n, plotted[:, :, ::-1].copy()))
                        best = sorted(best, key=lambda x: -x[0])[:4]
                if idx % 10 == 0:
                    bar.progress(min(idx / total, 1.0), text="Analyzing video...")
                idx += 1

            cap.release()
            writer.release()
            bar.progress(1.0, text="Done")

            # Fallback if the tracker produced no IDs: one summary row at the start
            if not events and (peak["crack"] or peak["hole"]):
                st.warning("Tracker gave no IDs; logged peak counts at the video start location.")
                add_logs(
                    [("video", vid.name, locator(0.0), v_source, peak["crack"], peak["hole"], None)]
                )
            else:
                add_logs(
                    [
                        (
                            "video",
                            vid.name,
                            loc,
                            v_source,
                            1 if k == "crack" else 0,
                            1 if k == "hole" else 0,
                            t,
                        )
                        for k, t, loc in events
                    ]
                )

            n_c = sum(1 for e in events if e[0] == "crack")
            n_h = sum(1 for e in events if e[0] == "hole")
            m1, m2, m3 = st.columns(3)
            m1.metric("Cracks (tracked)", n_c)
            m2.metric("Holes (tracked)", n_h)
            m3.metric("Most defects in one frame", max(peak.values()))

            if events:
                ev_df = pd.DataFrame(
                    [
                        {"type": k, "video_time_s": round(t, 1), "lat": loc[0], "lon": loc[1]}
                        for k, t, loc in events
                    ]
                )
                st.subheader("Each defect and where it was")
                st.dataframe(ev_df, use_container_width=True)
                st.map(ev_df[["lat", "lon"]])

            if best:
                st.subheader("Frames with the most defects")
                cols = st.columns(len(best))
                for col, (n, fr) in zip(cols, best):
                    col.image(fr, caption=f"{n} defect(s)", use_container_width=True)

            with open(out_path, "rb") as f:
                st.download_button("Download annotated video", f.read(), "annotated.mp4", "video/mp4")

            for p in (in_path, out_path):
                try:
                    os.remove(p)
                except OSError:
                    pass

# ------------------------------------------------------------- saved table
df = load_log()

if not df.empty:
    st.divider()
    st.subheader("📍 Locations ranked by cracks and holes")
    ranked = ranked_by_location(df)
    shown = ranked.drop(columns=["lat", "lon"])
    labels, n = [], 0
    with st.spinner("Looking up area names..."):
        for la_, lo_ in zip(ranked["lat"], ranked["lon"]):
            if pd.isna(la_) or pd.isna(lo_):
                labels.append(NO_LOCATION)
            else:
                n += 1
                fallback = f"Location {n}"
                # only look up the top 15 places to stay within the free service's limits
                labels.append(place_label(la_, lo_, fallback) if n <= 15 else fallback)
    shown.insert(0, "location", labels)
    st.dataframe(shown, use_container_width=True)

    with_loc = ranked.dropna(subset=["lat", "lon"])
    if not with_loc.empty:
        st.map(with_loc.assign(size=with_loc["total_defects"] * 5 + 10), size="size")

    missing = int(df["latitude"].isna().sum())
    if missing:
        st.caption(f"{missing} detection(s) have no location (shown in the row with empty lat/lon).")

    with st.expander("Full detection log"):
        st.dataframe(tidy_log(df), use_container_width=True)
        if st.checkbox("I want to delete the whole log", key="confirm_clear"):
            if st.button("Delete log", type="secondary"):
                clear_log()
                st.rerun()
