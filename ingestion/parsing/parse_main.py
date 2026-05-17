import os
import time
import json
import logging
from pathlib import Path
# from ingestion.parse.docling_parse import DoclingEngine
from ingestion.parsing.to_structured_html import HTMLCleaner

# 1. Initialize logger to track progress and errors
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():

    from ingestion.parsing.parse import DoclingEngine

    # 2. Define the source directory for input documents
    data_dir = Path("data/input_data")
    
    # 3. Automatically detect all PDF files in the target directory
    pdf_files = list(data_dir.glob("*.pdf"))

    # 4. Exit early if no documents are found to process
    if not pdf_files:
        logger.warning(f"No PDF files found in {data_dir}. Exiting.")
        return

    # 5. Set the root directories for output results and logs
    base_results_root = Path("ingestion_results/parsing_results")
    base_log_root = Path("logs/docling_parsing_log")

    # 6. Load the AI parsing engine with GPU acceleration enabled
    engine = DoclingEngine(use_gpu=True)

    # 7. Start the main loop to process each detected PDF
    for input_file in pdf_files:
        overall_start = time.time()
        
        # 8. Extract the filename without extension to use as a unique folder name
        file_slug = input_file.stem 
        
        # 9. Define specific output sub-directories for different file formats
        
        doc_output_root = base_results_root / file_slug
        raw_html_dir = doc_output_root / "html"
        structured_html_dir = doc_output_root / "structured" / "html"
     
        output_paths = {
            "markdown": doc_output_root / "markdown",
            "html": raw_html_dir,
            "json": doc_output_root / "json"
        }
        log_dir = base_log_root / file_slug
    
        # 10. Ensure all required output and log directories exist physically
      
        all_paths = [
            raw_html_dir, 
            structured_html_dir, 
            output_paths["markdown"], 
            output_paths["json"], 
            log_dir
        ]
        for path in all_paths:
            path.mkdir(parents=True, exist_ok=True)

        try:
            # 11. Notify that the AI conversion process has started
            logger.info(f"--- Starting AI parsing for: {input_file.name} ---")
            parse_start = time.time()
            
            # 12. Use the engine to convert the PDF into a structured document object
            doc = engine.convert_file(input_file)
            
            # 13. Record the number of pages and time taken for the heavy AI lift
            parse_duration = time.time() - parse_start
            num_pages = len(doc.pages)
            logger.info(f"AI Parsing complete. Pages: {num_pages} | Time: {parse_duration:.2f}s")

            # 14. Iterate through each page to extract specific format data
            for page_no in range(1, num_pages + 1):
                page_total_start = time.time()

                # 15. Generate Markdown content for the current page
                md_content = engine.get_markdown(doc, page_no)
                
                # 16. Generate HTML content for the current page
                html_content = engine.get_html(doc, page_no) 

                # 17. Save the Markdown output to the designated page file
                with open(output_paths["markdown"] / f"page_{page_no}.md", "w", encoding="utf-8") as f:
                    f.write(md_content)
                
                # 18. Save the HTML output to the designated page file
                with open(output_paths["html"] / f"page_{page_no}.html", "w", encoding="utf-8") as f:
                    f.write(html_content)

                # 19. Filter document elements to find items belonging to the current page
                page_elements = [e for e in doc.texts if e.prov and e.prov[0].page_no == page_no]
                
                # 20. Count text elements identified by the AI as headings
                headings = [e for e in page_elements if "heading" in (getattr(e, 'label', '') or '').lower()]
                
                # 21. Count complex table structures present on the current page
                tables = [t for t in doc.tables if t.prov and t.prov[0].page_no == page_no]

                # 22. Assemble metadata and performance stats for the current page
                page_log = {
                    "page_no": page_no,
                    "file_name": input_file.name,
                    "times": {
                        "total_page_sec": round(time.time() - page_total_start, 4)
                    },
                    "stats": {"headings": len(headings), "tables": len(tables)}
                }

                # 23. Write the page-specific metadata to a JSON log file
                with open(log_dir / f"log_page_{page_no}.json", "w", encoding="utf-8") as f:
                    json.dump(page_log, f, indent=4)

            # 24. Export the entire document's structured data into a master JSON
            json_dict = engine.get_json_data(doc)
            with open(output_paths["json"] / f"{file_slug}_full.json", "w", encoding="utf-8") as f:
                json.dump(json_dict, f)
            
            logger.info(f" Handing over to HTMLCleaner for: {file_slug}")
            cleaner = HTMLCleaner(input_dir=raw_html_dir, output_dir=structured_html_dir)

            cleaner.process_all(ordered=True)
            # 25. Log the final success message for the specific file
            logger.info(f"Finished {input_file.name} in {time.time() - overall_start:.2f}s")

        except Exception as e:
            logger.error(f"Error processing {input_file.name}: {e}")

if __name__ == "__main__":
 
    main()