# Retrieval_main.py
from query_retrieval.retrieval_engine import ExplainableTreeRAG
from query_retrieval.chat_history import SessionLogger

def main():
    print(" EXPLAINABLE TREE RAG ENGINE")
    print("   Vectorless • Hierarchical • Fully Transparent Retrieval")
    print("=" * 85)

    INDEX_PATH = "Ingestion_Results/Indexing_Results/Final_Indexing.json"

    # Initialize RAG and the new Logger
    rag = ExplainableTreeRAG(index_path=INDEX_PATH)
    logger = SessionLogger() # This creates the folder and the .txt file
    
    turn_count = 1

    print(f" Tree loaded successfully ({len(rag._title_tree.splitlines())} sections)")
    print(f" Logging session to: {logger.file_path}")
    print("   Ready for natural language queries.\n")

    while True:
        print("-" * 85)
        question = input(" Ask anything about the document: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            print("\n Goodbye!")
            break
        if not question:
            continue

        print("\n Searching with full tree reasoning...\n")

        # --- THE FIX IS HERE ---
        
        # 1. Run the query ONCE and store the output in 'result'
        result = rag.query(question)

        # 2. Log that result to your .txt file
        # We use .get() just in case the key is missing to avoid a crash
        logger.log_interaction(
            turn_number=turn_count, 
            question=question, 
            rewritten=result.get("rewritten_query", question), 
            answer=result.get("answer", "No answer found.")
        )

        # 3. Pass that SAME result to pretty_query to show it in the terminal
        # This prevents the LLM from running a second time!
        rag.pretty_query(question, result=result)
    

        turn_count += 1


if __name__ == "__main__":
    main()