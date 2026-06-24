import time
import streamlit as st

from report_qa_chat import ReportQASystem, ReportQAChat


def render_qa_chat_interface() -> None:
    st.subheader("Medical Report Q&A System")

    if "qa_system" not in st.session_state:
        st.session_state.qa_system = ReportQASystem(
            api_key=st.session_state.get("OPENROUTER_API_KEY")
        )
    if "qa_chat" not in st.session_state:
        st.session_state.qa_chat = ReportQAChat()
    if "qa_user_name" not in st.session_state:
        st.session_state.qa_user_name = "Dr. User"

    latest_key = st.session_state.get("OPENROUTER_API_KEY")
    if latest_key != st.session_state.qa_system.api_key:
        st.session_state.qa_system.api_key = latest_key

    if not latest_key:
        st.warning("Enter a valid OpenRouter API key in the sidebar to use Q&A.")

    user_name = st.text_input(
        "Your Name", value=st.session_state.qa_user_name, key="qa_name_input"
    )
    if user_name != st.session_state.qa_user_name:
        st.session_state.qa_user_name = user_name

    qa_tab1, qa_tab2 = st.tabs(["Join Existing Q&A", "Create New Q&A Room"])

    with qa_tab1:
        qa_rooms = st.session_state.qa_chat.get_qa_rooms()
        if qa_rooms:
            room_options = {
                f"{room['name']} (by {room['creator']})": room["id"]
                for room in qa_rooms
            }
            selected_room = st.selectbox(
                "Select Q&A Room",
                options=list(room_options.keys()),
                key="qa_room_select",
            )
            if st.button("Join Q&A Room", key="join_qa_btn"):
                st.session_state.current_qa_id = room_options[selected_room]
                st.rerun()
        else:
            st.info("No active Q&A rooms. Create a new one!")

    with qa_tab2:
        room_name = st.text_input("Q&A Room Name", key="qa_room_name_input")
        if st.button("Create Q&A Room", key="create_qa_btn"):
            if room_name:
                new_id = st.session_state.qa_chat.create_qa_room(user_name, room_name)
                st.session_state.current_qa_id = new_id
                st.rerun()
            else:
                st.error("Please provide a room name.")

    if "current_qa_id" not in st.session_state:
        return

    qa_id = st.session_state.current_qa_id
    qa_rooms = st.session_state.qa_chat.get_qa_rooms()
    current_room = next((r for r in qa_rooms if r["id"] == qa_id), None)

    if not current_room:
        st.error("This Q&A room no longer exists.")
        if st.button("Return to Room Selection", key="back_qa_btn"):
            del st.session_state.current_qa_id
            st.rerun()
        return

    st.subheader(f"Q&A Room: {current_room['name']}")
    st.caption(
        f"Created by {current_room['creator']} on {current_room['created_at'][:10]}"
    )

    if st.button("Clear Conversation History", key="clear_qa_hist"):
        st.session_state.qa_system.clear_history()
        st.info("Conversation history cleared.")

    messages = st.session_state.qa_chat.get_message(qa_id)
    with st.container():
        for msg in messages:
            is_ai = msg["user"] == "Report QA System"
            with st.chat_message(
                name=msg["user"], avatar="🤖" if is_ai else "👨‍💼"
            ):
                st.write(msg["content"])

    qa_message = st.chat_input(
        "Ask a question about your medical reports", key="qa_msg_input"
    )
    if qa_message:
        if not latest_key:
            st.warning("Please enter a valid API key in the sidebar first.")
        else:
            st.session_state.qa_chat.add_message(qa_id, user_name, qa_message)

            with st.spinner("Analysing medical reports…"):
                time.sleep(0.2)
                ai_response = st.session_state.qa_system.answer_question(qa_message)

            st.session_state.qa_chat.add_message(qa_id, "Report QA System", ai_response)
            st.rerun()

    with st.expander("Room Settings"):
        if st.button("Delete Q&A Room", key="del_qa_room"):
            if st.session_state.qa_chat.delete_qa_room(qa_id):
                st.success("Room deleted successfully.")
                del st.session_state.current_qa_id
                st.rerun()
            else:
                st.error("Failed to delete room.")
