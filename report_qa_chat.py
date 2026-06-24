import json
import logging
import math
import os
import uuid
from datetime import datetime

from openrouter_client import call_openrouter

logger = logging.getLogger(__name__)

_QA_STORE_PATH = "data/qa_chat_store.json"

_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "this", "that", "these", "those",
    "it", "its", "their", "there", "where", "which", "who", "what", "when",
    "how", "not", "no", "normal", "patient", "imaging", "image", "scan",
    "report", "finding", "findings", "shows", "noted", "seen", "mild",
    "moderate", "severe", "appears", "appear", "consistent", "noted",
}


# ReportQASystem -- answers questions using stored analyses as context

class ReportQASystem:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        self.conversation_history: list = []

    def _load_analysis_store(self) -> dict:
        """Always read fresh from disk so new analyses are picked up."""
        if os.path.exists("data/analysis_store.json"):
            try:
                with open("data/analysis_store.json", "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except json.JSONDecodeError:
                logger.warning("analysis_store.json is corrupt.")
        return {"analyses": []}

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lower-case, strip punctuation, remove stop words."""
        tokens = []
        for word in text.lower().split():
            word = word.strip(".,;:!?\"'()[]{}")
            if len(word) > 2 and word not in _STOP_WORDS:
                tokens.append(word)
        return tokens

    def get_relevant_contexts(self, query: str, top_k: int = 3) -> list:
        """
        Return the top-k most relevant analyses using TF-IDF-style scoring.

        Each analysis is scored by how many query terms appear in it,
        weighted by inverse document frequency so rare terms count more.
        """
        store = self._load_analysis_store()
        analyses = store.get("analyses", [])
        if not analyses:
            return ["No previous analyses found."]

        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return ["No valid analyses found."]

        doc_texts = []
        for analysis in analyses:
            full = analysis.get("analysis", "") + " " + " ".join(
                analysis.get("findings", [])
            )
            doc_texts.append(self._tokenize(full))

        N = len(doc_texts)
        idf: dict[str, float] = {}
        for token in query_tokens:
            df = sum(1 for doc in doc_texts if token in doc)
            idf[token] = math.log((N + 1) / (df + 1)) + 1.0 

        scored = []
        for idx, (analysis, tokens) in enumerate(zip(analyses, doc_texts)):
            token_set = set(tokens)
            score = sum(idf[t] for t in query_tokens if t in token_set)
            scored.append((score, analysis))

        scored.sort(key=lambda x: x[0], reverse=True)

        contexts = []
        for _, analysis in scored[:top_k]:
            chunk = analysis.get("analysis", "")
            if analysis.get("findings"):
                chunk += "\n\nFindings:\n" + "\n".join(
                    f"- {f}" for f in analysis["findings"]
                )
            chunk += f"\n\nImage: {analysis.get('filename', 'unknown')}"
            chunk += f"\nDate: {analysis.get('date', '')[:10]}"
            contexts.append(chunk)

        return contexts if contexts else ["No valid analyses found."]

    def answer_question(self, question: str) -> str:
        if not self.api_key:
            return "Please configure your OpenRouter API key to enable the Q&A system."

        contexts = self.get_relevant_contexts(question)
        combined_context = "\n\n---\n\n".join(contexts)

        system_prompt = (
            "You are a medical AI assistant answering questions about stored medical "
            "reports. Use the following context to answer accurately and concisely. "
            "If the answer cannot be found in the context, say so clearly.\n\n"
            f"Context:\n{combined_context}"
        )

        self.conversation_history.append({"role": "user", "content": question})

        try:
            messages = [{"role": "system", "content": system_prompt}] + self.conversation_history
            answer = call_openrouter(self.api_key, messages, max_tokens=600)
            self.conversation_history.append({"role": "assistant", "content": answer})

            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

            return answer

        except Exception as exc:
            self.conversation_history.pop()
            logger.error("answer_question failed: %s", exc)
            return f"I encountered an error while answering your question: {exc}"

    def clear_history(self) -> str:
        self.conversation_history = []
        return "Conversation history cleared."



# ReportQAChat 

class ReportQAChat:
    @staticmethod
    def _load() -> dict:
        if os.path.exists(_QA_STORE_PATH):
            try:
                with open(_QA_STORE_PATH, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except json.JSONDecodeError:
                logger.warning("qa_chat_store.json is corrupt -- resetting.")
        return {"rooms": {}}

    @staticmethod
    def _save(store: dict) -> None:
        tmp = _QA_STORE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(store, fh, indent=2)
        os.replace(tmp, _QA_STORE_PATH)

    def create_qa_room(self, user_name: str, room_name: str) -> str:
        store = self._load()
        room_id = f"QA-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:4]}"
        room_data = {
            "id": room_id,
            "name": room_name,
            "created_at": datetime.now().isoformat(),
            "creator": user_name,
            "messages": [
                {
                    "id": str(uuid.uuid4()),
                    "user": "Report QA System",
                    "content": (
                        f"Welcome to the Report Q&A room: {room_name}. "
                        "Ask questions about your medical reports and I'll answer "
                        "based on stored analyses."
                    ),
                    "timestamp": datetime.now().isoformat(),
                }
            ],
        }
        store["rooms"][room_id] = room_data
        self._save(store)
        return room_id

    def add_message(self, room_id: str, user_name: str, message: str) -> dict | None:
        store = self._load()
        if room_id not in store["rooms"]:
            return None
        message_data = {
            "id": str(uuid.uuid4()),
            "user": user_name,
            "content": message,
            "timestamp": datetime.now().isoformat(),
        }
        store["rooms"][room_id]["messages"].append(message_data)
        self._save(store)
        return message_data

    def get_message(self, room_id: str, limit: int = 50) -> list:
        store = self._load()
        if room_id not in store["rooms"]:
            return []
        messages = store["rooms"][room_id]["messages"]
        return messages[-limit:] if len(messages) > limit else messages

    def get_qa_rooms(self) -> list:
        store = self._load()
        rooms = [
            {
                "id": room_id,
                "name": data.get("name", "Unnamed Room"),
                "creator": data.get("creator", "Unknown"),
                "created_at": data.get("created_at", ""),
            }
            for room_id, data in store["rooms"].items()
        ]
        rooms.sort(key=lambda x: x["created_at"], reverse=True)
        return rooms

    def delete_qa_room(self, room_id: str) -> bool:
        store = self._load()
        if room_id in store["rooms"]:
            del store["rooms"][room_id]
            self._save(store)
            return True
        return False
