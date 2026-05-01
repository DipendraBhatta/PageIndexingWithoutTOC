import os
from dotenv import load_dotenv


from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

"""
CORE IDEAS:
1. NATIVE PRECISION: Uses ChatGroq directly to avoid LiteLLM dependency issues.
2. ENV VALIDATION: Strictly pulls configuration from .env for deployment safety.
3. GPS ANCHORING: Continues using XML tags to map text to physical pages.
"""

def get_llm():
    """
    Step 2: Factory to create the native LangChain ChatGroq object.
    """
    api_key = os.getenv("GROQ_API_KEY")
    # Step 3: Clean the model name from .env
    raw_model = os.getenv("LLM_MODEL")
    
    if not api_key:
        raise ValueError("[!] Error: GROQ_API_KEY not found in .env file.")
    if not raw_model:
        raise ValueError("[!] Error: LLM_MODEL not found in .env file.")
# Step 1: Use the Native LangChain-Groq driver
    # Step 4: Safety Strip - Removes 'groq/' if it exists to prevent 404 errors
    model_name = raw_model.replace("groq/", "").strip()

    return ChatGroq(
        groq_api_key=api_key,
        model=model_name,
        temperature=0.0,  # Factual precision for RAG indexing
        max_tokens=4096
    )

def call_llm(prompt: str, system_message: str = "You are a professional PDF analyzer."):
    """
    Step 5: Modular LangChain invocation using the .invoke() pattern.
    """
    llm = get_llm()
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=prompt)
    ]
    
    try:
        # Step 6: Standard LangChain response handling
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"[!] LangChain Native Error: {str(e)}"

def wrap_page_tags(page_text: str, page_number: int) -> str:
    """
    Step 7: The GPS Tagger. 
    Crucial for Vectorless RAG to track physical page locations.
    """
    return (
        f"<physical_index_{page_number}>\n"
        f"{page_text}\n"
        f"</physical_index_{page_number}>\n"
    )

# --- DIAGNOSTIC TEST ---
if __name__ == "__main__":
    print("[*] Diagnostic: Testing Pure LangChain Native Driver...")
    print(f"[*] Attempting to connect to model: {os.getenv('LLM_MODEL')}")
    
    test_reply = call_llm("Verify system: Reply with 'Native LangChain Online'.")
    print(f"\n[AI]: {test_reply}")