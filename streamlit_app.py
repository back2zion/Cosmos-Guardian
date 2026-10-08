
import streamlit as st
import tempfile
import cv2
import json
import time
from pathlib import Path
from PIL import Image

# Import the core agent
try:
    from core.agent import CosmosGuardianAgent
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from core.agent import CosmosGuardianAgent

st.set_page_config(
    page_title="Cosmos Guardian 🛡️",
    page_icon="🛡️",
    layout="wide"
)

# Initialize Session State for the Agent
if "agent" not in st.session_state:
    st.session_state.agent = None

def load_agent():
    with st.spinner("Initializing NVIDIA Cosmos Reason 2 Model (2B)..."):
        # We use a singleton pattern for the agent in session state
        # In a real deployed app, model loading should be cached differently (e.g. st.cache_resource)
        if st.session_state.agent is None:
            # Check for CUDA
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            st.session_state.agent = CosmosGuardianAgent(device=device)
            # Pre-load model to avoid delay on first run
            from core.agent import DEFAULT_ADAPTER_PATH
            st.session_state.agent.load_model(adapter_path=DEFAULT_ADAPTER_PATH)
            st.success(f"Model Loaded on {device.upper()}")

# --- Sidebar ---
with st.sidebar:
    st.title("🛡️ Cosmos Guardian")
    st.markdown("---")
    st.write("**Configuration**")
    
    model_id = st.selectbox(
        "Model Version",
        ["nvidia/Cosmos-Reason2-2B"],
        index=0
    )
    
    context = st.text_input(
        "Safety Context",
        value="Industrial Warehouse Safety",
        help="Define the environment context for the AI."
    )
    
    st.markdown("---")
    if st.button("Reload Model"):
        st.session_state.agent = None
        load_agent()
        
    st.markdown("---")
    st.info("Powered by NVIDIA Cosmos Reason 2")

# --- Main Content ---
st.title("Physical Safety Reasoning Agent")
st.markdown("Upload a video or image to detect hazards with **physics-aware logic**.")

# File Uploader
uploaded_file = st.file_uploader("Upload Media", type=["mp4", "avi", "mov", "jpg", "png", "jpeg"])

if uploaded_file:
    # Save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    media_path = tfile.name
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Media")
        if uploaded_file.type.startswith("video"):
            st.video(uploaded_file)
        else:
            st.image(uploaded_file)
            
    with col2:
        st.subheader("Live Reasoning Trace")
        
        if st.button("Start Analysis", type="primary"):
            if st.session_state.agent is None:
                load_agent()
            
            # Container for updates
            status_container = st.empty()
            progress_bar = st.progress(0)
            log_container = st.container()
            
            result_placeholder = st.empty()
            
            # Run Analysis
            agent = st.session_state.agent
            
            try:
                # Streaming generator
                full_reasoning = ""
                
                with log_container:
                    st.write("Processing started...")
                    
                    for update in agent.analyze_media(media_path, safety_context=context):
                        stage = update.get("stage")
                        
                        if stage == "model_loading":
                            status_container.info(update["detail"])
                        elif stage == "preprocessing":
                            status_container.info(update["detail"])
                            progress_bar.progress(10)
                        elif stage == "inference":
                            status_container.warning(update["detail"])
                            progress_bar.progress(30)
                        elif stage == "streaming_reasoning":
                            chunk = update.get("chunk", "")
                            # We don't want to re-render the whole chunk every time if it's huge, 
                            # but Streamlit captures stdout/write well.
                            # Just updating the status for now essentially.
                            if len(chunk) > len(full_reasoning):
                                new_text = chunk[len(full_reasoning):]
                                # st.write(new_text) # This might be too verbose
                                full_reasoning = chunk
                                status_container.markdown(f"**Generating Logic...**\n\n`{full_reasoning[-200:]}...`")
                                progress_bar.progress(60)

                        elif stage == "postprocessing":
                            status_container.success(update["detail"])
                            progress_bar.progress(90)
                        elif stage == "complete":
                            progress_bar.progress(100)
                            result = update["result"]
                            result_placeholder.json(result)
                            
                            # Visual Summary
                            if "hazards_detected" in result:
                                st.divider()
                                st.subheader("🛡️ Hazard Report")
                                for hazard in result["hazards_detected"]:
                                    severity = hazard.get("severity", "Medium")
                                    color = "red" if severity in ["High", "Critical"] else "orange"
                                    st.markdown(f"#### :{color}[{severity.upper()}] {hazard.get('type')}")
                                    st.write(f"**Description:** {hazard.get('description')}")
                                    st.info(f"**Physics Reasoning:** {hazard.get('reasoning')}")
                                    st.warning(f"**Prediction (2s):** {hazard.get('dynamic_prediction')}")
                                    st.markdown("---")
            
            except Exception as e:
                st.error(f"Analysis Failed: {e}")
                import traceback
                st.code(traceback.format_exc())

