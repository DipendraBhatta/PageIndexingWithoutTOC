import os
import subprocess
import logging
import sys
from pathlib import Path
import time

# --- LOGGER SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ingestion_results/indexing_results/pipeline_execution.log")
    ]
)
logger = logging.getLogger("Orchestrator")



# 1. Create the directory first
os.makedirs("ingestion_results/indexing_results", exist_ok=True)

def run_step(script_name, *args):
    """Utility to run a script and check for success."""
    logger.info(f"  EXECUTING: {script_name}")
    try:
        # 1. Copy the current environment (including .env variables if loaded)
        env = os.environ.copy()
        
        # 2. Tell the child process where the root directory is
        env["PYTHONPATH"] = os.getcwd() 

        # 3. Build the command
        command = [sys.executable, script_name] + [str(arg) for arg in args]
        
        # 4. Run it with the 'env' parameter
        result = subprocess.run(
            command, 
            check=True, 
            capture_output=True, 
            text=True, 
            env=env  # THIS IS THE CRITICAL LINE
        )
        logger.info(f"    {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"    CRITICAL ERROR in {script_name}")
        # This will show you exactly why the worker failed (e.g. missing API key)
        logger.error(f"    Error details: {e.stderr}")
        return False
def main():

    # --- CONFIGURATION ---
    print("\n" + "="*60)
    print("   AI DOCUMENT INDEXING PIPELINE  ")
    print("="*60)

    # --- AUTOMATIC SLUG SELECTION ---
    parsed_path = Path("ingestion_results/parsing_results")
    
    # 1. Get all directories inside parsing_results
    all_folders = [d for d in parsed_path.iterdir() if d.is_dir()]
    
    if not all_folders:
        logger.error(" No parsed documents found in 'ingestion_results/parsing_results'.")
        sys.exit(1)

    # 2. Sort by modification time (st_mtime) - newest first
    all_folders.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    
    # 3. Select the most recent one
    doc_slug = all_folders[0].name
    
    print(f"\n Auto-detected latest document: [{doc_slug}]")
    print(f" Last updated: {time.ctime(all_folders[0].stat().st_mtime)}")
    print("-" * 60)


    # --- PATH DEFINITIONS ---
    # 1. We look inside 'parsing_results' for the slug folder (e.g., 'sbc')
    BASE_INPUT = Path("ingestion_results/parsing_results") / doc_slug
    CONTENT_HTML = BASE_INPUT / "structured" / "html"


   # 2. We save everything into 'indexing_results' under the slug folder
    # We use parents=True to make sure the full path exists
    INDEXING_DIR = Path("ingestion_results/indexing_results") / doc_slug
    INDEXING_DIR.mkdir(parents=True, exist_ok=True)


    # 3. Setup Workspace for Step 2 (Root Node Summary debugs)
    # This folder is required by your root_node_summary script
    EXTRACTED_TEXT_DIR = Path("debug_verification/extracted_text")
    EXTRACTED_TEXT_DIR.mkdir(parents=True, exist_ok=True)


    PAGE_CONTENT_JSON = str(INDEXING_DIR / "Page_Indexing.json") 
    FINAL_OUTPUT = str(INDEXING_DIR / "Final_Indexing.json")
    SUMMARIZED_JSON = str(INDEXING_DIR / "Summarized_Nodes.json")

    # STEP 1: Content Extraction
    # Needs: Input HTML folder, Output JSON path, and the TOC pages to skip
    if not run_step("ingestion/indexing/semantic_page_extractor.py", CONTENT_HTML, PAGE_CONTENT_JSON):
        sys.exit(1)


    # STEP 2: ROOT NODE SUMMARY
    if not run_step("ingestion/indexing/root_node_summary.py", PAGE_CONTENT_JSON, CONTENT_HTML, EXTRACTED_TEXT_DIR, SUMMARIZED_JSON):
        sys.exit(1)

    # STEP 3: HIERARCHICAL AGGREGATION
    # Converts the flat summarized JSON into a nested Tree structure (Parent -> Child).
    # IMPORTANT: Reads SUMMARIZED_JSON so that LLM summaries are preserved!
    if not run_step("ingestion/indexing/indexing_aggregator.py", SUMMARIZED_JSON, FINAL_OUTPUT):
        sys.exit(1)


    print("\n" + "="*60)
    print("  SUCCESS! Your document has been perfectly indexed.")
    print(f" Output Location: {FINAL_OUTPUT}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()