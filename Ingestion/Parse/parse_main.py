import os
import time
import json
import logging
from pathlib import Path
from Ingestion.Parse.docling_parse import DoclingEngine

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    # --- 1. CONFIGURATION ---
    input_file = Path("Data/sbc.pdf")
    base_res_dir = Path("./Results")
    log_dir = Path("./LOG")
    
    output_paths = {
        "markdown": base_res_dir / "markdown",
        "html": base_res_dir / "html",
        "json": base_res_dir / "json"
    }
    
    for path in list(output_paths.values()) + [log_dir]:
        path.mkdir(parents=True, exist_ok=True)

    # --- 2. INITIALIZE ENGINE ---
    engine = DoclingEngine(use_gpu=True)
    overall_start = time.time()
    
    try:
        # --- 3. THE HEAVY LIFT (AI CONVERSION) ---
        logger.info(f"Starting AI parsing for: {input_file.name}")
        parse_start = time.time()
        
        doc = engine.convert_file(input_file)
        
        parse_duration = time.time() - parse_start
        num_pages = len(doc.pages)
        logger.info(f"AI Parsing complete. Total Pages: {num_pages} | Time: {parse_duration:.2f}s")

        # --- 4. INDIVIDUAL PAGE PROCESSING ---
        for page_no in range(1, num_pages + 1):
            page_total_start = time.time()

            # A. Export to Markdown
            t_md_start = time.time()
            md_content = engine.get_markdown(doc, page_no)
            t_md = time.time() - t_md_start

            # B. Export to HTML (Removed hybrid function, using standard HTML)
            t_html_start = time.time()
            html_content = engine.get_html(doc, page_no) 
            t_html = time.time() - t_html_start

            # --- 5. SAVING PAGE FILES ---
            with open(output_paths["markdown"] / f"page_{page_no}.md", "w", encoding="utf-8") as f:
                f.write(md_content)
            
            with open(output_paths["html"] / f"page_{page_no}.html", "w", encoding="utf-8") as f:
                f.write(html_content)

            # --- 6. METADATA LOGGING (PER PAGE) ---
            # Corrected text extraction for logging
            page_elements = [e for e in doc.texts if e.prov and e.prov[0].page_no == page_no]
            headings = [e for e in page_elements if "heading" in (getattr(e, 'label', '') or '').lower()]
            tables = [t for t in doc.tables if t.prov and t.prov[0].page_no == page_no]

            page_log = {
                "page_no": page_no,
                "individual_times_sec": {
                    "markdown_conversion": round(t_md, 4),
                    "html_conversion": round(t_html, 4),
                    "total_page_processing": round(time.time() - page_total_start, 4)
                },
                "content_stats": {
                    "headings": len(headings),
                    "tables": len(tables),
                    "char_count": len(html_content)
                }
            }

            with open(log_dir / f"log_page_{page_no}.json", "w", encoding="utf-8") as f:
                json.dump(page_log, f, indent=4)

            logger.info(f"Processed Page {page_no}/{num_pages}")

        # --- 7. SAVING MASTER JSON ---
        logger.info("Exporting Master JSON...")
        json_dict = engine.get_json_data(doc)
        with open(output_paths["json"] / "full_document.json", "w", encoding="utf-8") as f:
            json.dump(json_dict, f)
        
        total_time = time.time() - overall_start
        print("\n" + "="*50)
        print(f"PIPELINE SUMMARY")
        print(f"Total Pages: {num_pages} | Total Time: {total_time:.2f}s")
        print("="*50)

    except Exception as e:
        logger.error(f"Critical error in main pipeline: {e}", exc_info=True)

if __name__ == "__main__":
    main()