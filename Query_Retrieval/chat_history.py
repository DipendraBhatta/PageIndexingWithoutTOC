import os
from datetime import datetime

class SessionLogger:
    def __init__(self, folder_name="chat_history"):
        self.folder_name = folder_name
        os.makedirs(self.folder_name, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.file_path = os.path.join(self.folder_name, f"session_{timestamp}.txt")
        
        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write(f"=== SESSION STARTED: {timestamp} ===\n\n")

    def log_interaction(self, turn_number, question, rewritten, answer):
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(f"Turn {turn_number}\n")
            f.write(f"User Question: {question}\n")
            f.write(f"LLM Search Query: {rewritten}\n")
            f.write(f"Answer: {answer}\n")
            f.write("-" * 40 + "\n\n")