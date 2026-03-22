import json
import os
import uuid
from datetime import datetime
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ReportQASystem:
    def __init__(self,api_key=None):
        self.api_key=api_key
        self.conversation_history=[]
        self.analysis_store=self.load_analysis_store()


    def load_analysis_store(self):
        if os.path.exists("analysis_store.json"):
            with open("analysis_store.json","r") as f:
                return json.load(f)
        return {"analyses":[]}
    
    def get_embeddings(self,text,model="text-embedding-ada-002"):
        if not self.api_key:
            return np.random.rand(1536)
        
        try:
            client=openai.OpenAI(api_key=self.api_key)
            response=client.embeddings.create(
                input=text,
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embeddings: {e}")

            return np.random.rand(1536)
        


    def get_relevant_contexts(self, query, top_k=3):
        query_embedding = self.get_embeddings(query)

        analyses = self.analysis_store["analyses"]
        contexts = []

        if not analyses:
            return ["No previous analyses found."]

        # Build contexts
        for analysis in analyses:
            analysis_text = analysis.get("analysis", "")
            if not analysis_text.strip():
                continue

            full_text = analysis_text

            if "findings" in analysis and analysis["findings"]:
                findings_text = "\n".join([f"- {finding}" for finding in analysis["findings"]])
                full_text += f"\n\nFindings:\n{findings_text}"

            full_text += f"\n\nImage: {analysis.get('filename','unknown')}"
            full_text += f"\nDate: {analysis.get('date','')[:10]}"

            contexts.append({
                "text": full_text,
                "embedding": self.get_embeddings(full_text),
                "id": analysis.get("id", ""),
                "date": analysis.get("date", "")
            })

        if not contexts:
            return ["No valid analyses found."]

        similarities = []

        for context in contexts:
            similarity = cosine_similarity(
                [query_embedding],
                [context["embedding"]]
            )[0][0]

            similarities.append((similarity, context))

        similarities.sort(reverse=True)

        top_contexts = [context["text"] for _, context in similarities[:top_k]]

        return top_contexts
    


    def answer_question(self, question):
        try:
            client = openai.OpenAI(api_key=self.api_key)

            # Add user question to history
            self.conversation_history.append({
                "role": "user",
                "content": question
            })

            messages = [
                {"role": "system", "content": "You are a helpful medical report assistant."}
            ] + self.conversation_history

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=500,
                temperature=0.3
            )

            answer = response.choices[0].message.content

            # Save assistant reply
            self.conversation_history.append({
                "role": "assistant",
                "content": answer
            })

            # Keep only last 10 messages
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

            return answer

        except Exception as e:
            return f"I encountered an error while answering your question: {str(e)}"


    def clear_history(self):
        self.conversation_history=[]
        return "Conversation history is clear"
    

class ReportQAChat:
    def __init__(self):
        self.qa_chat_store=self.get_qa_chat_store()

    
    def get_qa_chat_store(self):
        """Get the QA chat storage"""
        if os.path.exists("qa_chat_store.json"):
            with open("qa_chat_store.json","r") as f:
                return json.load(f)
        return {"rooms":{}}
    
    def save_qa_chat_store(self):
        """Save the QA chat storage"""
        with open("qa_chat_store.json","w") as f:
            json.dump(self.qa_chat_store,f)

    def create_qa_room(self,user_name,room_name):
        """Create a new QA chat room"""
        room_id=f"QA-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        room_data={
            "id":room_id,
            "name":room_name,
            "created_at":datetime.now().isoformat(),
            "creator":user_name,
            "message":[]

        }
        welcome_message={
            "id":str(uuid.uuid4()),
            "user":"Report QA System",
            "content":f"Welcome to the Report QA room: {room_name}. You can ask question about your medical report and I'll try to answer based on the analyses stored in the system.",
            "timestamp":datetime.now().isoformat()
        }
        room_data["message"].append(welcome_message)

        self.qa_chat_store["rooms"][room_id]=room_data
        self.save_qa_chat_store()

        return room_id
    

    def add_message(self,room_id,user_name,message):
        """"Add a message to a QA room"""

        if room_id not in self.qa_chat_store["room"]:
            return None
        
        message_data={
            "id":str(uuid.uuid4()),
            "user":user_name,
            "content":message,
            "timestamp":datetime.now().isoformat()
        }

        self.qa_chat_store["room"][room_id]["message"].append(message_data)
        self.save_qa_chat_store()

        return message_data
    

    def get_message(self,room_id,limit=50):
        """Get the most recent message from a QA room"""
        if room_id not in self.qa_chat_store["rooms"]:
            return []
        
        message=self.qa_chat_store["room"][room_id]["message"]
        return message[-limit:] if len(message)>limit else message
    
    def get_qa_rooms(self):
        """Get all QA rooms"""
        rooms=[]
        for room_id, room_data in self.qa_chat_store["rooms"].items():
            rooms.append({
                "id":room_id,
                "name":room_data.get("name","Unnamed Room"),
                "creator":room_data.get("creator","Unknown"),
                "created_at":room_data.get("created_at","")
            })

        rooms.sort(key=lambda x: x["created_at"],reverse=True)
        return rooms
    

    def delete_qa_room(self,room_id):
        """Delete a QA chat room"""
        if room_id in self.qa_chat_store["rooms"]:
            del self.qa_chat_store["rooms"][room_id]
            self.save_qa_chat_store()
            return True
        return False
    


    
        