import os
import subprocess
import logging
import sys
from pathlib import Path

# --- LOGGER SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("Results/pipeline_execution.log")
    ]
)
logger = logging.getLogger("Orchestrator")

def run_step(script_name, *args):
    """Utility to run a script and check for success."""
    logger.info(f" EXECUTING: {script_name}")
    try:
        command = [sys.executable, script_name] + list(args)
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"  CRITICAL ERROR in {script_name}")
        logger.error(f"Error details: {e.stderr}")
        return False

# def ensure_dirs():
#     """Create required directories"""

os.makedirs("Indexing_Results", exist_ok=True)


def run_step(script_name, *args):
    """Utility to run a script and check for success."""
    logger.info(f" EXECUTING: {script_name}")
    try:
        command = [sys.executable, script_name] + list(args)
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"   {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"   CRITICAL ERROR in {script_name}")
        logger.error(f"    Error details: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error(f"   {script_name} not found. Create it first!")
        return False


def main():
    # --- BEAUTIFUL USER INTERACTION ---
    print("\n" + "="*60)
    print("   WELCOME TO THE AI DOCUMENT INDEXING PIPELINE  ")
    print("="*60)
    print("\nDear developer,")
    print("To ensure the highest accuracy for your RAG system, I need a small favor.")
    
    # Prompt the user for the TOC range
    toc_input = input("\n Please input the Table of Contents page numbers (e.g., '2' or '2,3,4'): ").strip()
    
    print(f"\nThank you! Your help will make the page indexing much better.")
    print("Starting the engine... check the logs for progress.\n")
    print("="*60 + "\n")


      # --- PATH DEFINITIONS ---
    CONTENT_HTML = "Ingestion_Results/Parsing_Results/structured/html"
    MARKDOWN_DIR = "Results/markdown"
    EXTRACTED_TEXT_DIR = "debug_verification/extracted_text"


    INDEXING_DIR = Path("Ingestion_Results/Indexing_Results")
    INDEXING_DIR.mkdir(parents=True, exist_ok=True)


    ASCII_TOC_PATH = str(INDEXING_DIR / "Markdown_to_ASCII_TOC.txt")
    TOC_SKELETON_JSON = str(INDEXING_DIR / "TOC_to_Indexing.json")
    PAGE_CONTENT_JSON = str(INDEXING_DIR / "page_indexing.json")
    FINAL_OUTPUT = str(INDEXING_DIR / "final_indexing.json")

    # --- EXECUTION FLOW ---

    # STEP 1: Content Extraction
    # Needs: Input HTML folder, Output JSON path, and the TOC pages to skip
    if not run_step("Ingestion/Indexing/semantic_page_extractor.py", CONTENT_HTML, PAGE_CONTENT_JSON, toc_input):
        sys.exit(1)

    # STEP 2: Markdown TOC
    # Needs: Input Markdown folder, Output TXT path, and the TOC pages to read
    # FIX: Added ASCII_TOC_PATH so the worker knows where to save!
    if not run_step("Ingestion/Indexing/markdown_toc.py", MARKDOWN_DIR, ASCII_TOC_PATH, toc_input):
        sys.exit(1)

    # STEP 3: Skeleton Generation
    # Needs: The TXT file from Step 2, and the JSON path to save the skeleton
    if not run_step("Ingestion/Indexing/toc_indexing.py", ASCII_TOC_PATH, TOC_SKELETON_JSON):
        sys.exit(1)

    # STEP 4: ROOT NODE SUMMARY
    if not run_step("Ingestion/Indexing/root_node_summary.py", PAGE_CONTENT_JSON, CONTENT_HTML, EXTRACTED_TEXT_DIR, FINAL_OUTPUT):
        sys.exit(1)

    # STEP 5: Final Aggregation
    # Needs: The individual page data, the skeleton hierarchy, and the final filename
    if not run_step("Ingestion/Indexing/Indexing_aggregator.py", PAGE_CONTENT_JSON, TOC_SKELETON_JSON, FINAL_OUTPUT):
        sys.exit(1)
    print("\n" + "="*60)
    print("  SUCCESS! Your document has been perfectly indexed.")
    print(" Check 'final_indexing.json' for the result.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()