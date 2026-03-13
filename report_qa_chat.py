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
        
    
    def get_relevant_contexts(self,query,top_k=3):
        query_embedding=self.get_embeddings(query)

        analyses=self.analysis_store["analyses"]
        contexts=[]

        if not analyses:
            return ["No previous analyses found."]
        
        for analysis in analyses:
            analysis_text=analysis.get("analysis","")
            if not analysis_text.strip():
                continue

            full_text=analysis_text
            if "finding" in analysis and analysis["findings"]:
                findings_text="\n".join([f"- {finding}" for finding in analysis["findings"]])
                full_text+=f"\n\nFindings:\n{findings_text}"

            full_text+=f"\n\nImage: {analysis.get('filename','unknown')}"
            full_text+=f"\nDate: {analysis.get('date','')[:10]}"

            contexts.append({
                "text":full_text,
                "embedding":self.get_embeddings(full_text),
                "id":analysis.get("id", ""),
                "date":analysis.get("date","")
            })

            similarities=[]
            for context in contexts:
                similarity=cosine_similarity(
                    [query_embedding],
                    [context["embedding"]]
                )[0][0]
                similarities.append((similarity,context))

            similarities.sort(reverse=True)
            top_contexts=[context["text"] for _, context in similarities[:top_k]]

            return top_contexts
        


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


    def clear_distory(self):
        self.conversation_history=[]
        return "Conversation history is clear"
    

class ReportQAChat:
    def __init__(self):
        self.qa_chat_store=self.get_qa_chat_store()
        