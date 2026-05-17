import subprocess
import sys
import logging
import time
import os
from pathlib import Path

class RAGPipelineOrchestrator:
    """
    Orchestrates the RAG ingestion process with updated folder names:
    Phase 1: Parsing (in ingestion/parsing)
    Phase 2: Indexing (in ingestion/indexing)
    """

    def __init__(self, log_file="pipeline_execution.log"):
        self.root_dir = os.getcwd()
        self.log_file = log_file
        self._setup_logger()

    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - [PIPELINE] - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.log_file)
            ]
        )
        self.logger = logging.getLogger("Orchestrator")

    def _execute_step(self, script_name, relative_dir):
        """
        Runs a script in isolation. This is essential to prevent the 
        name collision between your local files and system libraries.
        """
        target_dir = Path(self.root_dir) / relative_dir
        script_path = target_dir / script_name

        if not script_path.exists():
            self.logger.error(f"File not found: {script_path}")
            return False

        # Multi-line Python logic to clean the environment path
        python_logic = f"""
import sys
import os
target = r'{target_dir}'
script = r'{script_path}'
if target in sys.path:
    sys.path.remove(target)
sys.path.append(target)
with open(script, 'r', encoding='utf-8') as f:
    exec(f.read(), {{'__name__': '__main__', '__file__': script}})
"""

        try:
            self.logger.info(f"--- Starting: {script_name} ---")
            start_time = time.time()

            # Execute the script and wait for it to finish successfully
            subprocess.run(
                [sys.executable, "-c", python_logic],
                cwd=self.root_dir,
                check=True,
                env={**os.environ, "PYTHONPATH": self.root_dir}
            )

            duration = time.time() - start_time
            self.logger.info(f"Completed {script_name} in {duration:.2f}s")
            return True

        except subprocess.CalledProcessError:
            self.logger.error(f"Execution failed at {script_name}. Pipeline stopped.")
            return False

    def run(self):
        print("\n" + "="*60)
        print("         DOCUMENT INGESTION & INDEXING SYSTEM")
        print("="*60)

        # UPDATED WORKFLOW TO MATCH YOUR NEW FOLDER NAMES
        workflow = [
            ("parse_main.py", "ingestion/parsing"),
            ("indexing_main.py", "ingestion/indexing")
        ]

        for script, directory in workflow:
            if not self._execute_step(script, directory):
                return

        # Final success message with the specific output path
        print("\n" + "="*60)
        print("         SUCCESS: ALL PHASES COMPLETED")
        print("         Output: ingestion_results/indexing_results/Final_Indexing.json")
        print("="*60 + "\n")

if __name__ == "__main__":
    orchestrator = RAGPipelineOrchestrator()
    orchestrator.run()