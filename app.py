import streamlit as st
import os
import time
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from pipeline.engine import CortexEngine

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Cortex CEM | AI Engineering Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #262730;
    }
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: #1e1e1e;
        border: 1px solid #333;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #00d4ff;
    }
    .stButton>button {
        background-color: #00d4ff;
        color: black;
        border-radius: 5px;
        font-weight: bold;
    }
    /* Logo style */
    .logo-container {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "engine" not in st.session_state:
    with st.spinner("Initializing Cortex Engine..."):
        st.session_state.engine = CortexEngine(verbose=False)

engine = st.session_state.engine

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="logo-container">
        <h1 style='margin:0;'>🧠 CORTEX</h1>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("**Computational ORchestration for Technical Engineering eXecution**")
    st.divider()
    
    st.subheader("🛠️ Capabilities")
    st.info("""
    - **Thermal Analysis**: Heat conduction & convection
    - **Geometry Gen**: SDF-based 3D modeling
    - **Material RAG**: Physical property lookups
    - **RL Tuner**: Agentic performance optimization
    """)
    
    st.subheader("📚 Available Materials")
    materials = engine.list_materials()
    st.write(", ".join(materials))
    
    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main Header
st.title("🤖 Cortex Engineering Hub")
st.caption("Describe your engineering problem, and I'll simulate it for you.")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "plot" in message:
            st.pyplot(message["plot"])

# Chat Input
if prompt := st.chat_input("Ex: Design a heat sink for a 50W LED on a 5cm x 5cm aluminum plate"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process request
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Cortex is analyzing, solving, and validating..."):
            try:
                # Add delay to simulate thinking if needed, but the engine is usually fast enough
                result = engine.run(prompt)
                
                # Format summary into a nice chat response
                summary = result.summary()
                
                # Display Summary
                st.markdown("### 📊 Engineering Analysis Complete")
                st.code(summary, language="text")
                
                # If there's a temperature plot, generate and show it
                plot = None
                if result.problem_spec.problem_type == "thermal":
                    st.subheader("🌡️ Temperature Distribution")
                    fig = result.plot_temperature()
                    st.pyplot(fig)
                    plot = fig
                
                # If geometry was generated
                if result.geometry:
                    st.success("🏗️ 3D Geometry generated and exported to current directory!")
                    st.info(f"Geometry Type: {result.problem_spec.geometry_type}")
                
                # Save assistant response
                history_content = f"### Analysis for: {prompt}\n\n" + summary
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": history_content,
                    "plot": plot if plot else None
                })

            except Exception as e:
                error_msg = f"❌ **Error during execution:** {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

st.markdown("---")
st.caption("Powered by AgentScope RL & Cortex Physics Solvers")
