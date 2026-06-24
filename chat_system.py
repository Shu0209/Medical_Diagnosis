import json
import logging
import os
import time
import uuid
from datetime import datetime

import streamlit as st

from openrouter_client import call_openrouter

logger = logging.getLogger(__name__)

_CHAT_STORE_PATH = "data/chat_store.json"

DOCTOR_PERSONAS = {
    "Dr. Johnson": "You are Dr. Johnson, a cardiologist. Respond concisely from a cardiac perspective.",
    "Dr. Chen":    "You are Dr. Chen, a pulmonologist. Respond concisely from a pulmonary perspective.",
    "Dr. Patel":   "You are Dr. Patel, a radiologist. Respond concisely with radiological interpretation.",
}



# Chat store helpers


def _load_store() -> dict:
    if os.path.exists(_CHAT_STORE_PATH):
        try:
            with open(_CHAT_STORE_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError:
            logger.warning("chat_store.json is corrupt -- resetting.")
    return {"rooms": {}}


def _save_store(store: dict) -> None:
    tmp = _CHAT_STORE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(store, fh, indent=2)
    os.replace(tmp, _CHAT_STORE_PATH)


def get_chat_store() -> dict:
    return _load_store()


def save_chat_store(store: dict) -> None:
    _save_store(store)


def create_chat_room(case_id: str, creator_name: str, case_description: str) -> str:
    store = _load_store()
    if case_id not in store["rooms"]:
        room_data = {
            "id": case_id,
            "created_at": datetime.now().isoformat(),
            "creator": creator_name,
            "description": case_description,
            "participants": [
                creator_name,
                "Dr. Johnson", "Dr. Chen", "Dr. Patel",
            ],
            "messages": [
                {
                    "id": str(uuid.uuid4()),
                    "user": "System",
                    "content": (
                        f"Case discussion started for: '{case_description}'. "
                        "Invite colleagues to join and discuss the findings."
                    ),
                    "type": "text",
                    "timestamp": datetime.now().isoformat(),
                }
            ],
        }
        store["rooms"][case_id] = room_data
        _save_store(store)
    return case_id


def join_chat_room(case_id: str, user_name: str) -> bool:
    store = _load_store()
    if case_id in store["rooms"]:
        if user_name not in store["rooms"][case_id]["participants"]:
            store["rooms"][case_id]["participants"].append(user_name)
            _save_store(store)
        return True
    return False


def add_message(case_id: str, user_name: str, message: str, message_type: str = "text") -> dict | None:
    store = _load_store()
    if case_id not in store["rooms"]:
        return None
    msg = {
        "id": str(uuid.uuid4()),
        "user": user_name,
        "content": message,
        "type": message_type,
        "timestamp": datetime.now().isoformat(),
    }
    store["rooms"][case_id]["messages"].append(msg)
    _save_store(store)
    return msg


def get_messages(case_id: str, limit: int = 50) -> list:
    store = _load_store()
    if case_id not in store["rooms"]:
        return []
    msgs = store["rooms"][case_id]["messages"]
    return msgs[-limit:] if len(msgs) > limit else msgs


def get_available_rooms() -> list:
    store = _load_store()
    rooms = [
        {
            "id": room_id,
            "description": data["description"],
            "creator": data["creator"],
            "created_at": data["created_at"],
            "participants": len(data["participants"]),
        }
        for room_id, data in store["rooms"].items()
    ]
    rooms.sort(key=lambda x: x["created_at"], reverse=True)
    return rooms



# Doctor response helper


def get_doctor_response(
    doctor_name: str,
    user_question: str,
    case_description: str,
    findings: list | None = None,
    api_key: str | None = None,
) -> str:
    """Generate a response in the persona of the selected doctor."""
    if not api_key:
        return f"{doctor_name}: Please configure your OpenRouter API key in the sidebar."

    persona = DOCTOR_PERSONAS.get(
        doctor_name,
        f"You are {doctor_name}, a medical specialist. Respond concisely.",
    )
    findings_text = ""
    if findings:
        findings_text = "Key findings:\n" + "\n".join(f"- {f}" for f in findings)

    system_prompt = (
        f"{persona}\n"
        f'The case: "{case_description}". {findings_text}\n'
        "Give a concise specialist opinion."
    )

    try:
        return call_openrouter(
            api_key,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question},
            ],
            max_tokens=300,
        )
    except Exception as exc:
        logger.error("get_doctor_response failed: %s", exc)
        return f"{doctor_name}: I encountered an error: {exc}"



# Streamlit chat UI


def render_chat_interface() -> None:
    st.subheader("Multi-Doctor Collaboration")

    if "user_name" not in st.session_state:
        st.session_state.user_name = "Dr. Anonymous"

    user_name = st.text_input("Your Name", value=st.session_state.user_name)
    if user_name != st.session_state.user_name:
        st.session_state.user_name = user_name

    tab1, tab2 = st.tabs(["Join Existing Case", "Create New Case"])

    with tab1:
        rooms = get_available_rooms()
        if rooms:
            room_options = {
                f"{room['id']} -- {room['description']} (by {room['creator']})": room["id"]
                for room in rooms
            }
            selected_room = st.selectbox("Select Case", options=list(room_options.keys()))
            if st.button("Join Discussion"):
                selected_case_id = room_options[selected_room]
                if join_chat_room(selected_case_id, user_name):
                    st.session_state.current_case_id = selected_case_id
                    st.rerun()
        else:
            st.info("No active case discussions. Create a new one!")

    with tab2:
        case_description = st.text_input("Case Description")
        can_create = (
            st.session_state.get("file_data") is not None
            and st.session_state.get("file_type") is not None
        )
        if can_create:
            case_id = f"CHAT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            if st.button("Create Discussion"):
                if case_description:
                    create_case_id = create_chat_room(case_id, user_name, case_description)
                    st.session_state.current_case_id = create_case_id
                    st.rerun()
                else:
                    st.error("Please provide a case description.")
        else:
            st.info("Upload an image first to create a new case discussion.")

    if "current_case_id" not in st.session_state:
        return

    case_id = st.session_state.current_case_id
    store = _load_store()

    if case_id not in store["rooms"]:
        st.error("This case discussion no longer exists.")
        if st.button("Return to Room Selection"):
            del st.session_state.current_case_id
            st.rerun()
        return

    room_data = store["rooms"][case_id]
    st.subheader(f"Case Discussion: {room_data['description']}")
    st.caption(
        f"Created by {room_data['creator']} -- {len(room_data['participants'])} participants"
    )

    
    doctor_name: str | None = None
    use_doctor = st.checkbox("Get specialist opinion after my message", value=False)
    if use_doctor:
        raw = st.selectbox(
            "Select Specialist",
            ["Dr. Johnson (Cardiologist)", "Dr. Chen (Pulmonologist)", "Dr. Patel (Radiologist)"],
        )
        doctor_name = raw.split(" (")[0]

   
    with st.container():
        for msg in get_messages(case_id):
            is_system = msg["user"] == "System"
            avatar = "🏥" if is_system else ("👩‍💼" if msg["user"] == user_name else "👨‍💼")
            with st.chat_message(name=msg["user"], avatar=avatar):
                if msg.get("type") == "annotation":
                    st.write("**Image Annotation:**")
                st.write(msg["content"])

    
    user_message = st.chat_input("Type your message here")
    if user_message:
        add_message(case_id, user_name, user_message)

        if use_doctor and doctor_name:
            api_key = st.session_state.get("OPENROUTER_API_KEY")
            findings = st.session_state.get("findings")
            with st.spinner(f"{doctor_name} is typing…"):
                time.sleep(0.3)
                doc_resp = get_doctor_response(
                    doctor_name, user_message, room_data["description"], findings, api_key
                )
            add_message(case_id, doctor_name, doc_resp)

        st.rerun()

   
    with st.expander("Add Image Annotation"):
        annotation = st.text_area("Describe what you see in the image")
        if st.button("Submit Annotation") and annotation:
            add_message(case_id, user_name, annotation, message_type="annotation")
            st.rerun()


def create_manual_chat_room(creator_name: str, case_description: str) -> str:
    case_id = f"CHAT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return create_chat_room(case_id, creator_name, case_description)