import pandas as pd
import os
import logging
from Query_Retrieval.Retrieval_engine import ExplainableTreeRAG

class RAGEvaluator:

    def __init__(self, file_path, output_file, log_file, index_path):
        self.file_path = file_path
        self.output_file = output_file
        self.log_file = log_file
        self.index_path = index_path
        self.df = None
        self._setup_logger()
        self.rag = ExplainableTreeRAG(index_path=self.index_path)

    def _setup_logger(self):
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def normalize_headers(self):
        """Standardize column names to prevent mapping failures."""
        new_cols = []
        for i, (lvl0, lvl1) in enumerate(self.df.columns):
            l0, l1 = str(lvl0), str(lvl1)
            
            # Clean Level 0
            if "Unnamed" in l0:
                if 5 <= i <= 12: l0 = "Input_Tokens"
                elif 13 <= i <= 20: l0 = "Output_Tokens"
                else: l0 = "" 
            
            # Clean Level 1 (Remove 'Unnamed' and strip whitespace)
            if "Unnamed" in l1:
                l1 = ""
            else:
                l1 = l1.strip()
                
            new_cols.append((l0, l1))
        self.df.columns = pd.MultiIndex.from_tuples(new_cols)

   

    def capture_results(self, index, result):
        tracker = self.rag.cost_tracker
        report = tracker.get_report()

      
        
        print(f"Total calls: {len(tracker.calls)}")
        print(f"Total time: {report.get('total_llm_time_seconds', 0)}")
        print(f"Total input tokens: {report.get('total_input_tokens', 0)}")
        print(f"Total output tokens: {report.get('total_output_tokens', 0)}")
       

        # Get relevant fields for judging
        question_text = self.df.iloc[index, 2]
        actual_answer = self.df.iloc[index, 3]
        
        # FIXED: Capture predicted answer and check for 429/Empty failure
        predicted_answer = result.get("answer", "").strip()
        if not predicted_answer or "limit exceeded" in predicted_answer.lower():
            raise ConnectionError("Rate limit hit or empty response detected.")

        # Write Predicted Answer (Col 4) and Time (Col 21)
        self.df.iloc[index, 4] = predicted_answer
        self.df.iloc[index, 21] = report.get("total_llm_time_seconds", 0)


        # CALL LLM Checking for Accuracy Score (Col 22)
        print(f"Checking semantic accuracy...")
        score = self.llm_judge_score(question_text, actual_answer, predicted_answer)
        self.df.iloc[index, 22] = score
        print(f"   Score: {score}")

        # Dynamic Token Mapping
        for call in tracker.calls:
            step = str(call.step).strip()
            print(f"Step: {step}")
            print(f"   Input tokens: {call.input_tokens}")
            print(f"   Output tokens: {call.output_tokens}")

            # Match and Write Input Tokens
            if ("Input_Tokens", step) in self.df.columns:
                self.df.loc[index, ("Input_Tokens", step)] = call.input_tokens
                print(f"   Input ({step}) written to CSV")
            else:
                print(f"WARNING: ('Input_Tokens', '{step}') not found in columns!")

            # Match and Write Output Tokens
            if ("Output_Tokens", step) in self.df.columns:
                self.df.loc[index, ("Output_Tokens", step)] = call.output_tokens
                print(f"   Output ({step}) written to CSV")
            else:
                print(f"WARNING: ('Output_Tokens', '{step}') not found in columns!")



        tracker.reset()
        print("Tracker reset done\n")     


    def llm_judge_score(self, question, actual, predicted):
        """
        Uses a fast LLM to determine if the predicted answer matches the actual answer semantically.
        Returns 1 if correct, 0 if incorrect.
        """
        prompt = f"""
        You are an objective grading assistant.
        Compare the PREDICTED answer to the ACTUAL answer for the given QUESTION.

        CRITERIA:
        - If the PREDICTED answer conveys the same factual information as the ACTUAL answer, it is CORRECT (1).
        - If the PREDICTED answer contradicts the ACTUAL answer or misses the core fact, it is INCORRECT (0).
        - Be lenient with phrasing, synonyms, and formatting differences (e.g., "$500" vs "500 USD", "ten dollars" vs "$10").
        - Accept equivalent units or representations (e.g., "2,500" vs "2.5k", "twenty percent" vs "20%").
        - Be strict with core factual values: numbers, dates, names, and coverage limits must match.
        - Ignore minor wording differences like "covered" vs "included" if the meaning is the same.
        - If the predicted answer is incomplete but still contains the essential fact, mark it CORRECT (1).
        - If the predicted answer adds extra but non‑contradictory detail, still mark it CORRECT (1).
        - Respond with ONLY one character: 1 or 0. Do not include words, punctuation, or explanation.

        QUESTION: {question}
        ACTUAL ANSWER: {actual}
        PREDICTED ANSWER: {predicted}

        Respond with ONLY one character: 1 or 0.
        Do not include words, punctuation, or explanation.
        """

        try:
            # Use your RAG's LLM client with a fast model
            response = self.rag.llm.complete(prompt)
            score = response.text.strip()

            # Take only the first character to avoid verbose outputs
            if score and score[0] == "1":
                return 1
            else:
                return 0

        except Exception as e:
            print(f"Error during LLM judging: {e}")
            self.logger.error(f"LLM judging failed for question: {question} with error: {e}")
            return 0  # Default to incorrect if there's an error



    def run(self):
        # RESUME LOGIC: Load existing results if they exist, otherwise load input
        if os.path.exists(self.output_file):
            print(f" Found existing results file. Resuming from last checkpoint...")
            self.df = pd.read_csv(self.output_file, header=[0, 1])
        else:
            print(f" Starting fresh evaluation...")
            self.df = pd.read_csv(self.file_path, header=[0, 1])

        # Force all data to object type to allow mixed content writing
        self.df = self.df.astype(object) 
        self.normalize_headers()

        for index, row in self.df.iterrows():
            # FIXED: RESUME LOGIC: Check for NaN OR your sentinel value " "
            existing_ans = str(self.df.iloc[index, 4]).strip()
            if pd.notna(self.df.iloc[index, 4]) and existing_ans != "" and existing_ans != " ":
                continue
            question = self.df.iloc[index, 2] 
            if pd.isna(question):
                continue

            print(f"\n Processing Q#{index+1}: {question[:100]}...")
            try:
                result = self.rag.query(question)
                self.capture_results(index, result)
                # Save after every successful query
                self.df.to_csv(self.output_file, index=False)
            
            except (ConnectionError, Exception) as e:
                error_msg = str(e).lower()
                
                # Check for token/rate limit errors or our custom sentinel trigger
                if any(key in error_msg for key in ["limit exceeded", "quota", "rate limit", "429", "connectionerror"]):
                    print("\n" + "!"*60)
                    print(f" STOPPED: TOKEN LIMIT FINISHED OR RATE LIMIT REACHED at Q#{index+1}.")
                    print(f"Details: {e}")
                    
                    # FIXED: Apply sentinel value " " to all remaining rows
                    self.df.iloc[index:, 4] = " "
                    self.df.to_csv(self.output_file, index=False)
                    
                    print(f"Remaining rows marked with ' '. Progress saved. Restart script to resume.")
                    print("!"*60 + "\n")
                    
                    self.logger.error(f"Row {index} hit token limits: {e}")
                    break  # Exit the loop to stop processing
                
                else:
                    print(f"   Unexpected Error: {e}")
                    self.logger.error(f"Row {index} failed with error: {e}")

        print(f"\n Evaluation process ended. Results available in: {self.output_file}")
            


if __name__ == "__main__":
    CSV_IN = "Data/sbc_test.csv"
    CSV_OUT = "Data/TEST_RESULTS.csv"
    LOG = "evaluation.log"
    INDEX = "Ingestion_Results/Indexing_Results/Final_Indexing.json"

    evaluator = RAGEvaluator(CSV_IN, CSV_OUT, LOG, INDEX)
    evaluator.run()
    