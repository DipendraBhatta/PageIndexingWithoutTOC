import pandas as pd
import os
import logging
from Query_Retrieval.Retrieval_engine import ExplainableTreeRAG

class RAGEvaluator:
    def __init__(self, file_path, log_file, index_path):
        self.file_path = file_path
        self.log_file = log_file
        self.index_path = index_path
        self.df = None
        self._setup_logger()
        # Step 1: Initialize the RAG engine once
        self.rag = ExplainableTreeRAG(index_path=self.index_path)

    def _setup_logger(self):
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def run(self):
        # Step 2: Load CSV (header=[0,1] matches your 2-row header)
        self.df = pd.read_csv(self.file_path, header=[0, 1])
        
        # Step 3: Iterate and pass queries to the RAG
        for index, row in self.df.iterrows():
            question = row[('Question', 'Unnamed: 2_level_1')]
            if pd.isna(question): continue

            print(f" Processing Q#{index+1}: {question}")
            
            # --- THIS IS THE PROCESS BEGINNING ---
            # This is the exact same call used in Retrieval_main.py
            result = self.rag.query(question) 
            
            # Step 4: Extract metrics from the CostTracker
            self.capture_results(index, result)
            
        # Step 5: Save final state
        output_file = self.file_path.replace(".csv", "_RESULTS.csv")
        self.df.to_csv(output_file, index=False)
        print(f"Evaluation complete. Saved to: {output_file}")

    def capture_results(self, index, result):
        """Harvests data from the RAG's internal tracker."""
        report = self.rag.tracker.get_report()
        calls = self.rag.tracker.calls

        # Fill Predicted Answer & Total Time
        self.df.at[index, ('Predicted_Answer(Y_P)', 'Unnamed: 4_level_1')] = result.get("answer", "N/A")
        self.df.at[index, ('Total Time', 'Unnamed: 21_level_1')] = report.get('total_llm_time_seconds', 0)

        # Map steps to sub-columns
        step_map = {
            "domain_summary": "domain_summary",
            "query_rewrite": "query_rewrite",
            "relevance_check": "relevance_check",
            "query_classification": "query_classification",
            "choose_root": "choose_root",
            "choose_child": "choose_child",
            "extract_answer": "extract_answer",
            "synthesize_answer": "synthesize_answer"
        }

        # Fill Token Breakdown
        for call in calls:
            sub_col = step_map.get(call.step)
            if sub_col:
                self.df.at[index, ('Input_Tokens', sub_col)] = call.input_tokens
                self.df.at[index, ('Output_Tokens', sub_col)] = call.output_tokens

        # Reset tracker for the next question
        self.rag.tracker.reset()

if __name__ == "__main__":
    # Settings
    CSV_FILE = 'Data/sbc_test - Sheet.csv'
    LOG = 'evaluation.log'
    INDEX = 'Ingestion_Results/Indexing_Results/Final_Indexing.json'

    evaluator = RAGEvaluator(CSV_FILE, LOG, INDEX)
    evaluator.run()