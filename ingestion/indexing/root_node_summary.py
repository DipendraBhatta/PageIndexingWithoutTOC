import json
import re
import logging
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from ingestion.indexing.token_counter import TokenTracker

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("LLM_MODEL")

class RootNodeSummaryGenerator:
    """
    Surgically extracts structural markers and summarized topics to create 
    a high-precision routing map for the document's Root Node.
    """


    def __init__(self, structured_json_path: str):
        self.json_path = Path(structured_json_path)



        # Initialize Uniform Logger
        self._setup_logging()
        
        # Initialize Groq and Tracker
        self.tracker = TokenTracker(model_name=GROQ_MODEL)
        clean_model = GROQ_MODEL.replace("groq/", "").strip()
        self.llm = ChatGroq(api_key=GROQ_API_KEY, model_name=clean_model, temperature=0)

    def _setup_logging(self):
        self.logger = logging.getLogger("GeneratorLogger")
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        formatter = logging.Formatter("%(asctime)s - [GENERATOR] - %(message)s")
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        fh = logging.FileHandler("ingestion_results/indexing_results/generator_log.txt", encoding="utf-8")
        fh.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

    # Load the JSON nodes you created in the previous step
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.nodes = json.load(f)

    # The first node (index 0) is your Root; the rest are Page Nodes
        self.root_node = self.nodes[0]
        self.child_nodes = self.nodes[1:]


        print(f" Foundation Ready!")
        print(f"Document Root Title: {self.root_node.get('title')}")



    def build_page_range_map(self, html_folder_path: str):
        html_dir = Path(html_folder_path)
        page_map = []

        # Sort the HTML files numerically
        files = sorted(html_dir.glob("*.html"), key=lambda x: int(re.search(r'\d+', x.name).group()))
        for html_file in files:
            page_num = int(re.search(r'\d+', html_file.name).group())
            
            with open(html_file, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                
               # 1. Collect Headers (including Table Headers <th> and Bold <strong>)
                header_tags = soup.find_all(['h1', 'h2', 'h3', 'h4', 'strong', 'b', 'th'])
                headers = [h.get_text(strip=True) for h in header_tags if len(h.get_text(strip=True)) > 3]


                ## 2. Collect Paragraphs (Trimmed if > 100 chars: 1st sentence + 4 random)
                import random
                # 2. Collect Paragraphs (Logic: First 2  for paragraphs > 20 chars)
                p_tags = soup.find_all('p')
                paragraphs = [
                    " ".join((s := re.split(r'(?<=[\.\!\?])\s+', p_text))[:1] )
                    if len(p_text := p.get_text(strip=True)) > 50 and len(s := re.split(r'(?<=[\.\!\?])\s+', p_text)) > 3
                    else p.get_text(strip=True)
                    for p in p_tags if len(p.get_text(strip=True)) > 20
                ]
                                
                page_map.append({
                    "page": page_num,
                    "headers": headers,
                    "paragraphs": paragraphs
                    })
                    
        return page_map



    def collect_table_summaries(self,summary_folder_path: str):
        """
        Scans the table summaries folder and extracts headers/bold keys
        to identify what data is locked inside the tables.
        """
        summary_dir = Path(summary_folder_path)
        table_map = []

        # Filter for .md files and sort them by page number
        # Assumes filename format: table_page_1.md or similar
        files = sorted(summary_dir.glob("*.md"), 
                    key=lambda x: int(re.search(r'\d+', x.name).group()))

        for summary_file in files:
            page_num = int(re.search(r'\d+', summary_file.name).group())
            
            with open(summary_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Extract headers (#) and bold keys (**key**)
                # This captures the 'High Signal' parts of the table summary
                extracted_items = re.findall(r'(?:^|\n)(#{1,4}\s.*|\*\*.*?\*\*)', content)
                
                # Clean the findings
                cleaned_content = [item.strip() for item in extracted_items if len(item.strip()) > 2]

                if cleaned_content:
                    table_map.append({
                        "page": page_num,
                        "content": cleaned_content
                    })

        return table_map



    def collect_multi_table_topics(self, extracted_text_folder: str):
        """
        Parses summaries to extract ONLY the topic titles, 
        ignoring the 'Summary:' descriptions.
        """
        folder_path = Path(extracted_text_folder)
        results = []

        # Find all files and sort them numerically by page number
        files = sorted(folder_path.glob("*"), 
                       key=lambda x: int(re.search(r'\d+', x.name).group()) if re.search(r'\d+', x.name) else 0)

        for file_path in files:
            page_match = re.search(r'page_(\d+)', file_path.name)
            page_num = int(page_match.group(1)) if page_match else 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
               
                # We only process this file if it contains the multi-table marker
                if "PAGE_LEVEL_SUMMARY:" in content:
                    lines = content.split('\n')
                    topics = []
                    
                    for line in lines:
                        line = line.strip()
                        
                      
                        # Skip if line is empty OR contains "Summary:" OR "PAGE_LEVEL_SUMMARY:"
                        if not line or "Summary:" in line or "PAGE_LEVEL_SUMMARY:" in line:
                            continue
                        
                        # If the remaining line is a title (short enough), capture it
                        if 3 < len(line) < 100:
                            topics.append(line)
                    
                    if topics:
                        results.append({
                            "page": page_num,
                            "type": "multi-table", # Optional: label it
                            "content": topics
                        })
                else:
                    # Optional: Handle pages that are NOT multi-table here if needed
                    pass
                    
        return results




    def generate_master_page_map(self,html_map, table_map, multi_table_map):
        master_map = []
        
        # We use the html_map as our base (it has all page numbers 1-5)
        for html_page in html_map:
            page_num = html_page.get('page') or html_page.get('page_num')          
            # 1. Initialize the Master Entry for this page
            entry = {
                "page": page_num,
                "headers": html_page.get('headers', []),
                "paragraphs": html_page.get('paragraphs', []),
                "table_content": []
            }
            
            # 2. Check if this page is marked as MULTI-TABLE
            # (This comes from your Task 3.1 results)
            is_multi = next((m for m in multi_table_map if m['page'] == page_num), None)
            
            if is_multi:
                # PRIORITY: Use clean topics, SKIP the standard table map
                entry["table_content"] = is_multi.get('headers', [])
                # We can also add a flag for clarity
                entry["page_type"] = "multi_table_summary"
            else:
                # 3. If NOT multi-table, check for standard table content
                # (This comes from Table_Structural_Map.json)
                std_table = next((t for t in table_map if t['page'] == page_num), None)
                if std_table:
                    entry["table_content"] = std_table.get('content', [])
                    entry["page_type"] = "standard_with_table"
                else:
                    entry["page_type"] = "standard_narrative"

            master_map.append(entry)
            
        return master_map



    def generate_topic_centric_index(self, final_master_map: list):
        """
        Creates a new JSON output by combining topic titles from self.nodes
        with the surgical content stored in Final_Master_Map.
        """
        new_output = []

        # We use self.nodes which was loaded in __init__
        for entry in self.nodes:
            # 1. Skip the Root Title (Page 0)
            page_num_raw = entry.get("page_number")
            if page_num_raw == 0:
                continue

            topic_title = entry.get("title")
            
            # 2. Determine Page Range
            # We ensure start_page is treated as an integer for matching
            try:
                start_page = int(page_num_raw)
            except (ValueError, TypeError):
                continue

            # 3. Extract matching content from the master map we built earlier
            page_content = next((p for p in final_master_map if p["page"] == start_page), None)

            if page_content:
                # 4. Construct the organized routing data
                new_output.append({
                    "title": topic_title,
                    "page_number": str(start_page),
                    "content": {
                        "headers": page_content.get("headers", []),
                        "paragraphs": page_content.get("paragraphs", []),
                        "table_content": page_content.get("table_content", []),
                        "page_type": page_content.get("page_type", "standard")
                    }
                })

        return new_output

    


    def generate_llm_summary(self, routing_data: List[Dict]) -> str:
        """
        Generate a concise Root Summary from the routing map JSON.
        Focus: full summary 
        """

        # Build context bits: title + first 2 paragraphs + headers + table content
        context_bits = []
        for d in routing_data:
            title = d.get("title", "Untitled Section")
            page_num = d.get("page_number", "N/A")
            content = d.get("content", {})

            headers = content.get("headers", [])
            paragraphs = content.get("paragraphs", [])
            table_content = content.get("table_content", [])

            # Take first 2 paragraphs if available
            para_preview = " | ".join(paragraphs[:2]) if paragraphs else "No paragraphs"
            headers_preview = ", ".join(headers) if headers else "No headers"
            table_preview = ", ".join(table_content) if table_content else "No table content"

            context_bits.append(
                f"Page {page_num} - {title}\n"
                f"Headers: {headers_preview}\n"
                f"Paragraphs: {para_preview}\n"
                f"Table Content: {table_preview}\n"
            )

        # Build prompt for LLM
        prompt = f"""
        You are an expert document indexer.
        Task: Create a concise 'summary' from the routing map.

        Format:
        title: [Section Title]
        summary: [full  summary of the section based on headers, paragraphs, and table

       title: [Section Title]
       summary: [full  summary of the section based on headers, paragraphs, and table

    
        Data:
        {chr(10).join(context_bits)}
        """

        try:
            self.logger.info("Calling Groq for Root Summary...")
            response = self.llm.invoke(prompt)
            self.tracker.add_call(prompt, response.content)
            return response.content.strip()
        except Exception as e:
            self.logger.error(f"LLM Summary generation failed: {e}")
        return "Summary generation unavailable."





def main():
        # Style: [1]input_json, [2]html_dir, [3]text_dir, [4]output_final_json
        if len(sys.argv) < 5:
            print("\n Summary Generator Error: Missing arguments")
            print("Usage: python3 root_node_summary.py <input_json> <html_dir> <text_dir> <output_json>")
            sys.exit(1)

        # 1. Capture dynamic paths from Orchestrator
        json_path = sys.argv[1]
        html_folder = sys.argv[2]
        text_folder = sys.argv[3]
        output_path = Path(sys.argv[4])
        
        # This automatically finds your 'doc_slug' folder (e.g., .../indexing_results/sbc/)
        indexing_dir = output_path.parent 
        indexing_dir.mkdir(parents=True, exist_ok=True)

        # 2. Initialize Generator
        generator = RootNodeSummaryGenerator(json_path)

        # 3. HTML Processing -> Save Structural Map
        full_results = generator.build_page_range_map(html_folder)
        with open(indexing_dir / "Structural_Map.json", "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=4)

        # 4. Multi-Table Processing -> Save Multi-Table Topics
        multi_table_results = generator.collect_multi_table_topics(text_folder)
        with open(indexing_dir / "Multi_Table_Topics.json", "w", encoding="utf-8") as f:
            json.dump(multi_table_results, f, indent=4)

        # 5. Handle Table Structural Map (Check if exists)
        table_path = indexing_dir / "Table_Structural_Map.json"
        table_data = []
        if table_path.exists():
            with open(table_path, "r") as f:
                table_data = json.load(f)

        # 6. Generate Master Map -> Save Final Master Map
        final_master_json = generator.generate_master_page_map(full_results, table_data, multi_table_results)
        with open(indexing_dir / "FINAL_MASTER_MAP.json", "w", encoding="utf-8") as f:
            json.dump(final_master_json, f, indent=4)

        # 7. Generate Topic Centric Index -> Save Topic Routing Map
        final_organized_json = generator.generate_topic_centric_index(final_master_json)
        with open(indexing_dir / "TOPIC_ROUTING_MAP.json", "w", encoding="utf-8") as f:
            json.dump(final_organized_json, f, indent=4)

        # 8. Generate Summary and Inject into Root
        root_summary = generator.generate_llm_summary(final_organized_json)
        for entry in generator.nodes:
            if entry.get("page_number") == 0 or entry.get("node_id") == "0001":
                entry["merged_content"] = root_summary
                break

        # 9. Save Final Result (The one the Aggregator will use)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(generator.nodes, f, indent=4, ensure_ascii=False)

        generator.logger.info(f" SUCCESS: All {indexing_dir.name} debug files and final summary saved.")




if __name__ == "__main__":
     main()
    