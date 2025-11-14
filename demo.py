import streamlit as st
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent / "depth_data"

def load_array(path: Path):
    return np.load(path)

def list_classes():
    return sorted([d.name for d in BASE_DIR.iterdir() if d.is_dir()])

def get_metric_paths(class_name: str):
    class_dir = BASE_DIR / class_name

    da = None
    mp = None

    for p in class_dir.iterdir():
        if not p.is_file():
            continue
        name = p.name.lower()
        if name.startswith("metric_depth_depthanything"):
            da = p
        elif name.startswith("metric_depth_mediapipe"):
            mp = p

    return da, mp

def main():
    st.title("Depth Comparison Viewer (DepthAnything vs MediaPipe)")

    st.write("BASE_DIR:", BASE_DIR)

    classes = list_classes()
    if not classes:
        st.error(f"No class folders found in {BASE_DIR}")
        return

    class_name = st.selectbox("Class", classes)

    da_path, mp_path = get_metric_paths(class_name)

    st.write(f"DepthAnything file: `{da_path.name}`")
    st.write(f"MediaPipe file: `{mp_path.name}`")

    da_arr = load_array(da_path)
    mp_arr = load_array(mp_path)
    
    st.subheader("Comparison Array")
    st.line_chart(
        {"DepthAnything": da_arr, "MediaPipe": mp_arr}
    )

if __name__ == "__main__":
    main()