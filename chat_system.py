import streamlit as st
from datetime import datetime
import json
import os
import uuid
import time
import openai



def get_chat_store():
    if os.path.exists("chat_store.json"):
        with open("chat_store.json","r") as f:
            return json.load(f)
        
    return {"rooms": {}}


def save_chat_store(store):
    with open("chat_store.json","w") as f:
        json.dump(store, f)


def create_chat_room(case_id, creator_name, case_description):
    store=get_chat_store()

    if case_id not in store["rooms"]:
        room_data={
            "id":case_id,
            "create_at":datetime.now().isoformat(),
            "creator":creator_name,
            "description":case_description,
            "participants":[creator_name, "Dr. AI Assistant", "Dr. Johnson","Dr. Chen", "Dr. Patel"],
            "message":[]
        }

        #Welcome Message
        welcome_message={
            "id":str(uuid.uuid4()),
            "user":"Dr. AI Assistant",
            "content":f"Welcome to the case discussion for '{case_description}'. I've analyzed the image and I'm here to asist with the diagnosis. Feel free to ask me specific questions about the findings.",
            "type":"text",
            "timestamp":datetime.now().isoformat()
        }
        room_data["message"].append(welcome_message)

        store["room"][case_id]=room_data
        save_chat_store(store)

    return case_id



def join_chat_room(case_id, user_name):
    store=get_chat_store()

    if case_id in store["rooms"]:
        if user_name not in store["rooms"][case_id]["participants"]:
            store["rooms"][case_id]["participants"].append(user_name)
            save_chat_store(store)
        return True
    return False


def add_message(case_id, user_name, message, message_type="text"):
    store=get_chat_store()

    if case_id in store["rooms"]:
        message_data={
            "id":str(uuid.uuid4()),
            "user":user_name,
            "content":message,
            "type":message_type,
            "timestamp":datetime.now().isoformat()
        }
        store["rooms"][case_id]["message"].append(message_data)
        save_chat_store(store)
        return message_data
    
    return None

def get_messages(case_id,limit=50):
    store=get_chat_store()

    if case_id in store["rooms"]:
        message=store["rooms"][case_id]["messages"]
        return message[-limit:] if len(message)>limit else message
    
    return []

def get_available_rooms():
    store=get_chat_store()
    rooms=[]

    for room_id, room_data in store["rooms"].items():
        rooms.append({
            "id":room_id,
            "description":room_data["description"],
            "creator":room_data["creator"],
            "created_at":room_data["created_at"],
            "participants":len(room_data["participants"])
        })

    rooms.sort(key=lambda x: x["created_at"], reverse=True)
    return rooms


def get_openai_response(user_question, case_description, findings=None,api_key=None):
    if not api_key:
        return "Please configure your OpenAI API key in the sidebar to get AI responses."
    
    #Set ip OpenAI client
    client=openai.OpenAI(api_key=api_key)

    #Create the findings text if available
    findings_text=""
    if findings and len(findings)>0:
        findings_text="The key findings in the image are:\n"

    # Create system prompt with medical context
    system_prompt=f"""You are Dr. AI Assistant, a medical specialist analyzing a medical image.
    The image is from a case described as:"{case_description}".
    {findings_text}
    Please provide an expert, accurate, and helpful response to the doctor's question.
    Base your response on the findings and your medical expertise.
    Respond as if you are speaking directly to the doctor in a collaborative setting.
    Keep your response concise but informative,focused on the relevant medical details.
    """

    try:
        #Make API call
        response=client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role":"system","content":system_prompt},
                {"role":"user","content":user_question}
            ],
            max_tokens=300,
            temperature=0.2
        )

        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return f"I apologize, but I encountered an error while anlyzing your questions. Please try again or rephrase your question. Error details: {str(e)}"


    