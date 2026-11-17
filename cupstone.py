import io
import os
import tempfile
from datetime import datetime

import cv2
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="Crack Detection", page_icon="🔍", layout="wide")

MODEL_PATH = "best .pt"  



@st.cache_resource
def load_model(path: str):
    from ultralytics import YOLO
    return YOLO(path)


try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load `{MODEL_PATH}`: {e}")
    st.stop()

class_names = list(model.names.values())



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


def pick_location(key: str, exif_loc=None):
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
    st.info("📍 Allow location access in your browser to attach a location...")
    return None, None


def is_selected(cls_id: int) -> bool:
    return model.names[int(cls_id)].lower() in selected_lower


def count_selected(result) -> int:
    if result.boxes is None or len(result.boxes) == 0:
        return 0
    return sum(is_selected(c) for c in result.boxes.cls.int().tolist())


def add_log(kind, name, location, source, cracks):
    st.session_state.log.append(
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": kind,
            "file": name,
            "latitude": location[0] if location else None,
            "longitude": location[1] if location else None,
            "location_source": source,
            "cracks_found": cracks,
        }
    )


def ranked_by_location(df: pd.DataFrame) -> pd.DataFrame:
    """Group by location (~11 m grid) and sort so the location with the most cracks is first."""
    d = df.dropna(subset=["latitude", "longitude"]).copy()
    if d.empty:
        return d
    d["lat"] = d["latitude"].round(4)
    d["lon"] = d["longitude"].round(4)
    g = d.groupby(["lat", "lon"], as_index=False).agg(
        total_cracks=("cracks_found", "sum"),
        detections=("file", "count"),
        last_detection=("time", "max"),
    )
    g = g.sort_values("total_cracks", ascending=False).reset_index(drop=True)
    g.index = g.index + 1
    g.index.name = "rank"
    return g


if "log" not in st.session_state:
    st.session_state.log = []


conf = 0.25      
imgsz = 640      
stride = 3       

selected_lower = {c.lower() for c in class_names if "crack" in c.lower()} or {
    c.lower() for c in class_names
}

st.title("🔍 Crack Detection")
st.caption("Upload an image or a video, set where it was captured, and detect cracks.")

tab_img, tab_vid = st.tabs(["📷 Image", "🎥 Video"])


with tab_img:
    mode = st.radio("Image source", ["Upload photo", "Take photo"], horizontal=True)
    file = (
        st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="img_up")
        if mode == "Upload photo"
        else st.camera_input("Capture an image")
    )

    if file:
        raw = Image.open(io.BytesIO(file.getvalue()))
        gps, taken = read_exif(raw) if mode == "Upload photo" else (None, None)
        img = ImageOps.exif_transpose(raw).convert("RGB")

        if not gps:
            st.warning("No GPS data in this image (it is often stripped by browsers and chat apps).")
        location, source = pick_location("img", gps)

        left, right = st.columns(2)
        left.image(img, caption="Input image", use_container_width=True)
        with right:
            if location:
                st.write(f"📍 **{location[0]:.6f}, {location[1]:.6f}** ({source})")
                if taken:
                    st.write(f"**Captured:** {taken}")
                st.map(pd.DataFrame({"lat": [location[0]], "lon": [location[1]]}), zoom=14)

        if st.button("Run detection", type="primary", key="img_run"):
            with st.spinner("Detecting cracks..."):
                res = model.predict(img, conf=conf, imgsz=imgsz, verbose=False)[0]
                annotated = res.plot()[:, :, ::-1]
                n = count_selected(res)
            st.image(annotated, caption=f"{n} crack(s) detected", use_container_width=True)
            add_log("image", getattr(file, "name", "camera.jpg"), location, source, n)


with tab_vid:
    st.info(
        "Videos rarely carry GPS data, so set one location for the whole video. "
        "Cracks are tracked across frames so the same crack is not counted twice."
    )
    vid = st.file_uploader("Choose a video", type=["mp4", "mov", "avi", "mkv"], key="vid_up")

    if vid:
        v_location, v_source = pick_location("vid")
        st.video(vid)

        if st.button("Run detection on video", type="primary", key="vid_run"):
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
            track_ids, peak, best, idx = set(), 0, [], 0
            bar = st.progress(0.0, text="Analyzing video...")

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % stride == 0:
                    res = model.track(
                        frame, persist=True, conf=conf, imgsz=imgsz, verbose=False
                    )[0]
                    n = count_selected(res)
                    peak = max(peak, n)
                    if res.boxes is not None and res.boxes.id is not None:
                        for tid, c in zip(res.boxes.id.int().tolist(), res.boxes.cls.int().tolist()):
                            if is_selected(c):
                                track_ids.add(tid)
                    plotted = res.plot()
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

            cracks = len(track_ids) if track_ids else peak
            c1, c2 = st.columns(2)
            c1.metric("Cracks (tracked across frames)", cracks)
            c2.metric("Most cracks in a single frame", peak)

            if best:
                st.subheader("Frames with the most cracks")
                cols = st.columns(len(best))
                for col, (n, fr) in zip(cols, best):
                    col.image(fr, caption=f"{n} crack(s)", use_container_width=True)

            with open(out_path, "rb") as f:
                st.download_button(
                    "Download annotated video", f.read(), "annotated.mp4", "video/mp4"
                )
            add_log("video", vid.name, v_location, v_source, cracks)

            for p in (in_path, out_path):
                try:
                    os.remove(p)
                except OSError:
                    pass


if st.session_state.log:
    st.divider()
    df = pd.DataFrame(st.session_state.log)

    st.subheader("📍 Locations ranked by number of cracks")
    ranked = ranked_by_location(df)
    if ranked.empty:
        st.info("No detections with a location yet.")
    else:
        st.dataframe(ranked, use_container_width=True)
        st.map(ranked.assign(size=ranked["total_cracks"] * 5 + 10), size="size")

    missing = df["latitude"].isna().sum()
    if missing:
        st.caption(f"{missing} detection(s) have no location and are not ranked above.")

    with st.expander("Full detection log"):
        st.dataframe(df, use_container_width=True)
