from dotenv import load_dotenv
import pandas as pd
import os
import logging
import time
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

from query_retrieval.retrieval_engine import ExplainableTreeRAG


class RAGEvaluator:
        def __init__(self, file_path, output_file, log_file, index_path):
            self.file_path = file_path
            self.output_file = output_file
            self.log_file = log_file
            self.index_path = index_path
            self.df = None
            
            self._setup_logger()
            self.rag = ExplainableTreeRAG(index_path=self.index_path)
            
            # === Judge LLM (Separate from main RAG model) ===
            self.judge_llm = ChatOpenAI(
                model="anthropic/claude-3-haiku",   # Fast, cheap, and follows instructions well
                openai_api_key=os.getenv("OPEN_ROUTER_API_KEY"),
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0,
                default_headers={
                    "HTTP-Referer": "http://localhost:3000",
                    "X-Title": "RAG Evaluation Script",
                },
            )

        def _setup_logger(self):
            logging.basicConfig(
                filename=self.log_file,
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s"
            )
            self.logger = logging.getLogger(__name__)

        def normalize_headers(self):
            """Standardize column names"""
            new_cols = []
            for i, (lvl0, lvl1) in enumerate(self.df.columns):
                l0, l1 = str(lvl0).strip(), str(lvl1).strip()
                
                if "Unnamed" in l0 or l0 == "":
                    if 5 <= i <= 12:
                        l0 = "Input_Tokens"
                    elif 13 <= i <= 20:
                        l0 = "Output_Tokens"
                    elif i == 21:
                        l0 = "Total_Input_Tokens"
                    elif i == 22:
                        l0 = "Total_Output_Tokens"
                    elif i == 23:
                        l0 = "Total Time"
                    elif i == 24:
                        l0 = "Evaluation"
                        
              # Remove _o suffix for Output_Tokens
                if l1.endswith('_o'):
                            l1 = l1[:-2]   # remove _o
                new_cols.append((l0, l1))
                            
            self.df.columns = pd.MultiIndex.from_tuples(new_cols)

            print("\n=== FINAL COLUMN MAPPING ===")
            print("Input_Tokens columns :", [col[1] for col in self.df.columns if col[0] == "Input_Tokens"])
            print("Output_Tokens columns:", [col[1] for col in self.df.columns if col[0] == "Output_Tokens"])
            print("Total columns        :", [col for col in self.df.columns if col[0] in ["Total_Input_Tokens", "Total_Output_Tokens", "Total Time", "Evaluation"]])
            print("="*70)


        def llm_judge_score(self, question, actual, predicted):
            """Improved LLM Judge"""
            prompt = f"""
                You are an objective grading assistant.

                QUESTION: {question}
                ACTUAL ANSWER: {actual}
                PREDICTED ANSWER: {predicted}

                Evaluate if the PREDICTED answer is semantically correct compared to the ACTUAL answer.
                - Same core fact (especially numbers, amounts, names) → 1
                - Different or missing core fact → 0
                - Be lenient with wording and format

                Respond with ONLY one character: 1 or 0
                """

            try:
                time.sleep(0.8)  # To avoid hitting rate limits
                
                response = self.judge_llm.invoke([HumanMessage(content=prompt)])
                raw_output = response.content.strip()
                
                print(f"Judge Raw Output: '{raw_output}'")
                
                # Robust parsing
                if '1' in raw_output:
                    return 1
                else:
                    return 0
                    
            except Exception as e:
                print(f"Error during LLM judging: {e}")
                self.logger.error(f"LLM judging failed: {e}")
                return 0

        def capture_results(self, index, result):
            tracker = self.rag.cost_tracker
            report = tracker.get_report()

            question_text = self.df.iloc[index, 2]
            actual_answer = self.df.iloc[index, 3]
            predicted_answer = result.get("answer", "").strip()

            if not predicted_answer or "limit exceeded" in predicted_answer.lower():
                raise ConnectionError("Rate limit hit or empty response detected.")

            # Save Predicted Answer and Time
            self.df.iloc[index, 4] = predicted_answer                          # Predicted Answer
            self.df.iloc[index, 21] = report.get("total_input_tokens", 0)      # Total_Input_Tokens
            self.df.iloc[index, 22] = report.get("total_output_tokens", 0)     # Total_Output_Tokens
            self.df.iloc[index, 23] = report.get("total_llm_time_seconds", 0)  # Total Time
                  
                  
            # Judge
            print(f"Checking semantic accuracy...")
            score = self.llm_judge_score(question_text, actual_answer, predicted_answer)
            self.df.iloc[index, 24] = score
            print(f" Score: {score}")

            # Token logging
            for call in tracker.calls:
                step = str(call.step).strip()
                
                # Input Tokens
                for col_idx, col in enumerate(self.df.columns):
                    if col[0] == "Input_Tokens" and col[1] == step:
                        self.df.iloc[index, col_idx] = call.input_tokens
                        break
                
                # Output Tokens (after removing _o in normalize_headers)
                for col_idx, col in enumerate(self.df.columns):
                    if col[0] == "Output_Tokens" and col[1] == step:
                        self.df.iloc[index, col_idx] = call.output_tokens
                        break
            tracker.reset()
            print("Tracker reset done\n")

        def run(self):
            # Load or resume
            if os.path.exists(self.output_file):
                print(f" Found existing results file. Resuming...")
                self.df = pd.read_csv(self.output_file, header=[0, 1])
            else:
                print(f" Starting fresh evaluation...")
                self.df = pd.read_csv(self.file_path, header=[0, 1])

            self.df = self.df.astype(object)
            self.normalize_headers()

            resumed = False
            start_index = 0

            
            # Find the first row that needs processing
            for i in range(len(self.df)):
                pred = str(self.df.iloc[i, 4]).strip()
                if pred in ["", " ", "nan", "NaN"]:
                    start_index = i
                    resumed = True
                    break
                    
                    
                else:
                    # All rows already processed
                    print("All questions have been evaluated already!")
                    print(f"Results available in: {self.output_file}")
                    return
                


            if resumed:
                     print(f" Resuming from Question #{start_index + 1}\n")

                
                # Start processing from the first incomplete row
            for index in range(start_index, len(self.df)):
                    question = self.df.iloc[index, 2]
                    if pd.isna(question) or str(question).strip() == "":
                        continue
            
                    print(f"\n Processing Q#{index+1}: {str(question)[:150]}...")

                    try:
                        result = self.rag.query(question)
                        self.capture_results(index, result)
                        
                        # Save after every successful evaluation
                        self.df.to_csv(self.output_file, index=False)
                        print(f" Saved progress at Q#{index+1}")

                    except Exception as e:
                        error_msg = str(e).lower()
                        if any(k in error_msg for k in ["limit", "429", "quota", "rate", "connectionerror"]):
                            print("\n" + "!"*70)
                            print(f" RATE LIMIT / TOKEN LIMIT REACHED at Q#{index+1}")
                            print("Remaining rows marked. Restart script to continue.")
                            print("!"*70)
                            
                            # Mark remaining rows as pending
                            self.df.iloc[index:, 4] = " "
                            self.df.to_csv(self.output_file, index=False)
                            break
                        else:
                            print(f" Unexpected Error at Q#{index+1}: {e}")
                            self.logger.error(f"Row {index} failed: {e}")
                            continue

                    print(f"\n Evaluation process ended. Results available in: {self.output_file}")
                                            


if __name__ == "__main__":
        CSV_IN = "Data/sbc_test(10).csv"
        CSV_OUT = "Data/TEST_RESULTS(2).csv"
        LOG = "evaluation.log"
        
        INDEX = "Ingestion_Results/Indexing_Results/Final_Indexing.json"

        evaluator = RAGEvaluator(CSV_IN, CSV_OUT, LOG, INDEX)
        evaluator.run()