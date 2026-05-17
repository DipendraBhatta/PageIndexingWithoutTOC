import tiktoken

class TokenTracker:
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        # We use cl100k_base encoding which is standard for modern LLMs
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
            
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0

    def _count(self, text: str) -> int:
        if not text: return 0
        return len(self.encoding.encode(text))

    def add_call(self, prompt: str, response: str):
        in_t = self._count(prompt)
        out_t = self._count(response)
        
        self.total_input_tokens += in_t
        self.total_output_tokens += out_t
        self.call_count += 1
        return in_t, out_t

    def get_report(self):
        return {
            "total_calls": self.call_count,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens
        }