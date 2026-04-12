import streamlit as st
import uuid
import time
from langchain_core.messages import HumanMessage, AIMessage

from agent.graph import compiled_graph
from agent.state import LeadProfile
from memory.db import init_db

# Ensure the DB is initialized
init_db()

# ─── Streamlit Page Config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoStream Aaru - AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("AutoStream · Aaru Agent")
st.markdown("Your personalized AI video platform assistant.")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "tool_activated" not in st.session_state:
    st.session_state.tool_activated = False


with st.sidebar:
    st.header("Under the Hood")
    intent_placeholder = st.empty()
    conf_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    st.divider()
    st.subheader("Lead Profile")
    profile_placeholder = st.empty()


for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg["content"])
            
            if msg.get("lead_captured") and not st.session_state.tool_activated:
                st.success("Lead Captured Tool Activated & Executed Successfully!")
                st.session_state.tool_activated = True

if prompt := st.chat_input("Message Aaru..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
     
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
        
    config = {"configurable": {"thread_id": st.session_state.session_id}}
    input_state = {
        "messages": [HumanMessage(content=prompt)],
        "session_id": st.session_state.session_id,
    }
    with st.spinner("Aaru is typing..."):
        try:
            final_state = compiled_graph.invoke(input_state, config=config)
        except Exception as e:
            st.error(f"Error processing message: {e}")
            st.stop()

    ai_response = ""
    for msg in reversed(final_state.get("messages", [])):
        if isinstance(msg, AIMessage):
            ai_response = msg.content
            break

    intent = final_state.get("last_intent", "Unknown")
    confidence = final_state.get("last_confidence", 0.0)
    reasoning = final_state.get("last_reasoning", "None")
    lead_captured = final_state.get("lead_captured", False)
    profile: LeadProfile = final_state.get("lead_profile", LeadProfile())
    turn = final_state.get("turn_count", 0)

    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(ai_response)
        if lead_captured and not st.session_state.tool_activated:
            st.success("Lead Captured Tool Activated & Executed Successfully! Backend notified.", icon="🚀")
            st.session_state.tool_activated = True

    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_response,
        "lead_captured": lead_captured
    })

    intent_placeholder.metric("Current Intent", str(intent))
    conf_placeholder.progress(confidence, text=f"Confidence: {confidence:.2f}")
    stats_placeholder.caption(f"*Reasoning:* {reasoning}\n\n*Turns:* {turn}")


    profile_html = ""
    for field in ["name", "email", "platform"]:
        val = getattr(profile, field, None)
        if val:
            profile_html += f"**{field.capitalize()}**: {val}<br>"
        else:
            profile_html += f"⏳ *{field.capitalize()}*: Missing<br>"
            
    profile_placeholder.markdown(profile_html, unsafe_allow_html=True)
