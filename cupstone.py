import io
import json
import os
import re
import sqlite3
import tempfile
import time
import xml.etree.ElementTree as ET
from contextlib import closing
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps

st.set_page_config(page_title="Crack Detection", page_icon="🔍", layout="wide")

MODEL_PATH = "best_pretrained.pt"
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detections.db")
NO_LOCATION = "No given location"



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
    """rows: list of (kind, name, location, source, cracks, video_time)"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = [
        (now, k, n, loc[0] if loc else None, loc[1] if loc else None, s, int(c), 0, vt)
        for k, n, loc, s, c, vt in rows
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


def add_log(kind, name, location, source, cracks):
    add_logs([(kind, name, location, source, cracks, None)])


def load_log() -> pd.DataFrame:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        return pd.read_sql_query(
            "SELECT time, type, file, latitude, longitude, location_source, "
            "cracks_found, video_time FROM detections ORDER BY id",
            conn,
        )


def clear_log():
    with closing(sqlite3.connect(DB_PATH)) as conn:
        with conn:
            conn.execute("DELETE FROM detections")


init_db()



@st.cache_resource
def load_model(path: str):
    from ultralytics import YOLO
    return YOLO(path)


try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load `{MODEL_PATH}`: {e}")
    st.stop()

# Only the crack classes are used; any other class (e.g. holes) is ignored.
CRACK_IDS = [int(i) for i, n in model.names.items() if "crack" in n.lower()]
if not CRACK_IDS:
    st.error(f"No class with 'crack' in its name was found in the model. Classes: {model.names}")
    st.stop()


def count_cracks(result) -> int:
    return 0 if result.boxes is None else len(result.boxes)


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


def device_location(key: str, quiet=False):
    """(lat, lon) from the browser/device location, or None while waiting."""
    try:
        from streamlit_js_eval import get_geolocation
    except ImportError:
        st.error("Install `streamlit-js-eval` (pip install streamlit-js-eval) to get the device location.")
        return None
    loc = get_geolocation(component_key=f"{key}_geo")
    if loc and "coords" in loc:
        return loc["coords"]["latitude"], loc["coords"]["longitude"]
    if not quiet:
        st.info("📍 Allow location access in your browser to attach a location...")
    return None


def video_metadata_location(data: bytes):
    """GPS stored by the phone inside the video file (QuickTime/MP4 '©xyz' atom).
    This is ONE point for the whole video (usually where recording started)."""
    i = data.find(b"\xa9xyz")
    if i == -1:
        return None
    m = re.search(rb"([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)", data[i + 8:i + 80])
    if not m:
        return None
    lat, lon = float(m.group(1)), float(m.group(2))
    if abs(lat) > 90 or abs(lon) > 180 or (lat == 0 and lon == 0):
        return None
    return lat, lon


def parse_gpx(data: bytes):
    """Return a sorted list of (unix_time, lat, lon) from a GPX track."""
    root = ET.fromstring(data)
    pts = []
    for el in root.iter():
        if el.tag.endswith("trkpt"):
            t = None
            for ch in el:
                if ch.tag.endswith("time") and ch.text:
                    t = datetime.fromisoformat(ch.text.strip().replace("Z", "+00:00"))
            if t is not None:
                pts.append((t.timestamp(), float(el.get("lat")), float(el.get("lon"))))
    pts.sort()
    return pts


def gpx_locator(pts, offset_s: float):
    """Function t_seconds_in_video -> (lat, lon). The video start is matched to the
    first GPX point (+ offset)."""
    ts = np.array([p[0] for p in pts])
    la = np.array([p[1] for p in pts])
    lo = np.array([p[2] for p in pts])
    t0 = ts[0]

    def f(t):
        x = t0 + t + offset_s
        return float(np.interp(x, ts, la)), float(np.interp(x, ts, lo))

    return f


def tidy_log(df: pd.DataFrame) -> pd.DataFrame:
    """Replace empty values (None/NaN) with a dash so the log table looks clean."""
    d = df.copy()
    d["video_time"] = d["video_time"].map(lambda v: f"{v:.1f} s" if pd.notna(v) else "-")
    for c in ("latitude", "longitude"):
        d[c] = d[c].map(lambda v: f"{v:.6f}" if pd.notna(v) else NO_LOCATION)
    d["location_source"] = d["location_source"].fillna(NO_LOCATION)
    return d


@st.cache_data(show_spinner=False, ttl=86400)
def governorate_name(lat: float, lon: float) -> str:
    """Governorate (state-level region) for a coordinate (OpenStreetMap Nominatim, English if available).
    Raises on network errors so a failed lookup is not cached. Nominatim allows ~1 request/second."""
    time.sleep(1.1)
    r = requests.get(
        "https://nominatim.openstreetmap.org/reverse",
        params={"format": "jsonv2", "lat": lat, "lon": lon, "zoom": 8,
                "addressdetails": 1, "accept-language": "en"},
        headers={"User-Agent": "crack-detection-streamlit-app"},
        timeout=10,
    )
    r.raise_for_status()
    a = r.json().get("address", {})
    return (a.get("state") or a.get("state_district") or a.get("region")
            or a.get("county") or "")


def governorate_label(lat, lon, fallback: str) -> str:
    try:
        return governorate_name(float(lat), float(lon)) or fallback
    except Exception:
        return fallback



TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
TILE_ATTR = "Tiles &copy; Esri"


def show_map(points, height: int = 400):
    """points: list of dicts with lat, lon, r (marker radius in px) and optional label."""
    if not points:
        return
    html = f"""
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <div id="map" style="height:{height}px;border-radius:8px"></div>
    <script>
      const pts = {json.dumps(points)};
      const map = L.map('map');
      L.tileLayer('{TILE_URL}', {{maxZoom: 19, attribution: '{TILE_ATTR}'}}).addTo(map);
      const bounds = [];
      pts.forEach(p => {{
        const m = L.circleMarker([p.lat, p.lon],
          {{radius: p.r, color: '#d62728', fillColor: '#d62728', fillOpacity: 0.6}}).addTo(map);
        if (p.label) m.bindTooltip(p.label);
        bounds.push([p.lat, p.lon]);
      }});
      if (bounds.length === 1) map.setView(bounds[0], 16);
      else map.fitBounds(bounds, {{padding: [30, 30]}});
    </script>
    """
    components.html(html, height=height + 10)



def ranked_by_governorate(df: pd.DataFrame):
    """Name each place with its governorate and rank governorates by cracks.
    Returns (ranked table, grid points for the map)."""
    d = df.copy()
    d["lat"] = d["latitude"].round(4)  # ~11 m grid, used for the map
    d["lon"] = d["longitude"].round(4)
    g = d.groupby(["lat", "lon"], as_index=False, dropna=False).agg(
        total_cracks=("cracks_found", "sum"),
        last_detection=("time", "max"),
    )
    g = g.sort_values("total_cracks", ascending=False).reset_index(drop=True)

    
    names, labels = {}, []
    for la, lo in zip(g["lat"], g["lon"]):
        if pd.isna(la) or pd.isna(lo):
            labels.append(NO_LOCATION)
            continue
        cell = (round(la, 2), round(lo, 2))
        if cell not in names:
            names[cell] = governorate_label(la, lo, "Unknown governorate") if len(names) < 30 else "Other"
        labels.append(names[cell])
    g.insert(0, "governorate", labels)

    table = g.groupby("governorate", as_index=False).agg(
        total_cracks=("total_cracks", "sum"),
        last_detection=("last_detection", "max"),
    )
    table = table.sort_values("total_cracks", ascending=False).reset_index(drop=True)
    table.index = table.index + 1
    table.index.name = "rank"
    return table, g



imgsz = 640
stride = 3

conf = 0.40        
iou = 0.10          
contain_thr = 0.50  

st.title("🔍 Crack Detection")
st.caption("Upload an image or a video, set where it was captured, and detect cracks.")

tab_img, tab_vid = st.tabs(["📷 Image", "🎥 Video"])



with tab_img:
    mode = st.radio("Image source", ["Upload photo", "Take photo"], horizontal=True)
    if mode == "Upload photo":
        
        file = st.file_uploader(
            "Choose an image", type=["jpg", "jpeg", "png", "webp"], key="img_up"
        )
    else:
        cam_col, _ = st.columns([1, 2])  
        with cam_col:
            file = st.camera_input("Capture an image")

    if file:
        raw = Image.open(io.BytesIO(file.getvalue()))
        gps, taken = read_exif(raw) if mode == "Upload photo" else (None, None)
        img = ImageOps.exif_transpose(raw).convert("RGB")

        if gps:
            location, source = gps, "Photo EXIF"
        else:
            location = device_location("img", quiet=True)
            source = "Device location" if location else None

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
                show_map([{"lat": location[0], "lon": location[1], "r": 8}], height=300)

        if st.button("Run detection", type="primary", key="img_run"):
            with st.spinner("Detecting..."):
                res = model.predict(
                    img, conf=conf, iou=iou, agnostic_nms=True, imgsz=imgsz,
                    classes=CRACK_IDS, verbose=False
                )[0]
                res = remove_overlaps(res, contain_thr)
                annotated = res.plot(line_width=2, font_size=10)[:, :, ::-1]
                n_cracks = count_cracks(res)
            st.image(annotated, caption=f"{n_cracks} crack(s) detected", use_container_width=True)
            add_log("image", getattr(file, "name", "camera.jpg"), location, source, n_cracks)



with tab_vid:
    st.info(
        "Each crack is tracked across frames and logged separately with its own location. "
        "Location priority: GPX track (a different location per crack) → GPS saved inside the "
        "video file (one location for the whole video) → device location."
    )
    vid = st.file_uploader("Choose a video", type=["mp4", "mov", "avi", "mkv"], key="vid_up")
    gpx_file = st.file_uploader(
        "Optional: GPX track recorded while filming (gives each crack its own location)",
        type=["gpx"],
        key="gpx_up",
    )
    offset = st.number_input(
        "GPX offset in seconds (use if the GPX recording started before/after the video)",
        value=0.0, step=1.0,
    )

    if vid:
        st.video(vid)

        locator, v_source = None, None

        if gpx_file:
            try:
                pts = parse_gpx(gpx_file.getvalue())
                if len(pts) < 2:
                    st.error("The GPX file needs at least 2 track points with timestamps.")
                else:
                    locator, v_source = gpx_locator(pts, offset), "GPX track"
                    st.success(f"GPX loaded: {len(pts)} points.")
            except Exception as e:
                st.error(f"Could not read the GPX file: {e}")

        if locator is None:
            meta = video_metadata_location(vid.getvalue())
            if meta:
                locator, v_source = (lambda t, f=meta: f), "Video metadata"
                st.success(f"GPS found inside the video: {meta[0]:.6f}, {meta[1]:.6f} "
                           "(one location for all cracks).")

        if locator is None:
            dev = device_location("vid")
            if dev:
                locator, v_source = (lambda t, f=dev: f), "Device location"
            else:
                st.warning("No GPX or video GPS found. Waiting for the device location. "
                           "Allow location access in your browser.")

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

            model.predictor = None  
            seen = set() 
            events = []  
            peak = 0
            best, idx = [], 0
            bar = st.progress(0.0, text="Analyzing video...")

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % stride == 0:
                    t_sec = idx / fps
                    res = model.track(
                        frame, persist=True, conf=conf, iou=iou, agnostic_nms=True,
                        imgsz=imgsz, classes=CRACK_IDS, verbose=False
                    )[0]
                    res = remove_overlaps(res, contain_thr)
                    n = count_cracks(res)
                    peak = max(peak, n)

                    if res.boxes is not None and res.boxes.id is not None:
                        for tid in res.boxes.id.int().tolist():
                            if tid not in seen:
                                seen.add(tid)
                                events.append((t_sec, locator(t_sec)))

                    plotted = res.plot(line_width=2, font_size=10)
                    writer.write(plotted)
                    if n > 0:
                        best.append((n, plotted[:, :, ::-1].copy()))
                        best = sorted(best, key=lambda x: -x[0])[:4]
                if idx % 10 == 0:
                    bar.progress(min(idx / total, 1.0), text="Analyzing video...")
                idx += 1

            cap.release()
            writer.release()
            bar.progress(1.0, text="Done")

            if not events and peak:
                st.warning("Tracker gave no IDs; logged the peak count at the video start location.")
                add_logs([("video", vid.name, locator(0.0), v_source, peak, None)])
            else:
                add_logs([("video", vid.name, loc, v_source, 1, t) for t, loc in events])

            m1, m2 = st.columns(2)
            m1.metric("Cracks (tracked)", len(events))
            m2.metric("Most cracks in one frame", peak)

            if events:
                ev_df = pd.DataFrame(
                    [
                        {"crack": i, "video_time_s": round(t, 1), "lat": loc[0], "lon": loc[1]}
                        for i, (t, loc) in enumerate(events, start=1)
                    ]
                )
                st.subheader("Each crack and where it was")
                st.dataframe(ev_df, use_container_width=True, hide_index=True)
                show_map([{"lat": r.lat, "lon": r.lon, "r": 7, "label": f"Crack {r.crack} ({r.video_time_s} s)"}
                          for r in ev_df.itertuples()])

            if best:
                st.subheader("Frames with the most cracks")
                cols = st.columns(len(best))
                for col, (n, fr) in zip(cols, best):
                    col.image(fr, caption=f"{n} crack(s)", use_container_width=True)

            with open(out_path, "rb") as f:
                st.download_button("Download annotated video", f.read(), "annotated.mp4", "video/mp4")

            for p in (in_path, out_path):
                try:
                    os.remove(p)
                except OSError:
                    pass



df = load_log()

if not df.empty:
    st.divider()
    st.subheader("📍 Governorates ranked by cracks")
    with st.spinner("Looking up governorates..."):
        ranked, grid = ranked_by_governorate(df)
    st.dataframe(ranked, use_container_width=True)

    with_loc = grid.dropna(subset=["lat", "lon"])
    show_map(
        [
            {"lat": r.lat, "lon": r.lon, "r": min(8 + r.total_cracks * 2, 30),
             "label": f"{r.governorate}: {int(r.total_cracks)} crack(s)"}
            for r in with_loc.itertuples()
        ]
    )

    missing = int(df["latitude"].isna().sum())
    if missing:
        st.caption(f"{missing} detection(s) have no location (shown as '{NO_LOCATION}').")

    with st.expander("Full detection log"):
        st.dataframe(tidy_log(df), use_container_width=True)
        if st.checkbox("I want to delete the whole log", key="confirm_clear"):
            if st.button("Delete log", type="secondary"):
                clear_log()
                st.rerun()
