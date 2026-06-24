import streamlit as st

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["name"]):
        st.write(msg["content"])

# Input
prompt = st.chat_input("Type your message...")

if prompt:
    # User message
    st.session_state.messages.append({
        "name": "Shubham",
        "content": prompt
    })

    # Bot response
    response = "This is a reply"
    st.session_state.messages.append({
        "name": "Assistant",
        "content": response
    })