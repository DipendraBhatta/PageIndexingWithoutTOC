import sys
import re
import json
import os
import logging
from pathlib import Path
from typing import List
from bs4 import BeautifulSoup
from Ingestion.Indexing.page_schema import DocumentIndexFactory
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from Ingestion.Indexing.token_counter import TokenTracker

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("LLM_MODEL")

class PageExtractor:
    def __init__(self, input_path, output_path):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.factory = DocumentIndexFactory()
        self.tracker = TokenTracker(model_name=GROQ_MODEL)

        # Debug
        self.debug_dir = Path("debug_verification")
        self.page_out = self.debug_dir / "extracted_text"
        self.table_out = self.debug_dir / "table_summaries"
        self.page_out.mkdir(parents=True, exist_ok=True)
        self.table_out.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        clean_model = GROQ_MODEL.replace("groq/", "").strip()
        self.llm = ChatGroq(api_key=GROQ_API_KEY, model_name=clean_model, temperature=0)
        self.all_files = self._get_sorted_files()

    def _setup_logging(self):
        self.logger = logging.getLogger("IndexingLogger")
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        formatter = logging.Formatter("%(asctime)s - [INDEXING] - %(message)s")
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        fh = logging.FileHandler("Indexing_log.txt", encoding="utf-8")
        fh.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

    def _get_sorted_files(self):
        if not self.input_path.exists():
            self.logger.warning(f"Input path does not exist: {self.input_path}")
            return []
        files = list(self.input_path.rglob("page_*.html"))
        files.sort(key=lambda x: int(re.search(r'\d+', x.name).group()))
        return files

    def _clean_text(self, text: str) -> str:
        if not text: return ""
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
        return " ".join(text.split())

    def _extract_table_rows(self, table_html: str) -> str:
        soup = BeautifulSoup(table_html, 'parser.html') if 'parser.html' in str(BeautifulSoup) else BeautifulSoup(table_html, 'html.parser')
        rows = soup.find_all('tr')
        return "".join(str(row) for row in rows)

    def _stitch_tables(self, first_table_html: str, second_table_html: str) -> str:
        soup_main = BeautifulSoup(first_table_html, 'html.parser')
        target_table = soup_main.find('table')
        if not target_table: return first_table_html
        rows_to_add = self._extract_table_rows(second_table_html)
        new_rows_soup = BeautifulSoup(rows_to_add, 'html.parser')
        first_headers = [th.get_text(strip=True) for th in target_table.find_all('th')]
        for tr in new_rows_soup.find_all('tr'):
            if any(th.get_text(strip=True) in first_headers for th in tr.find_all('th')):
                tr.decompose()
        container = target_table.find('tbody') if target_table.find('tbody') else target_table
        container.append(new_rows_soup)
        return str(target_table)

    def _is_table_split(self, current_soup, next_file_path):
        """
        A table is likely split if the table on the next page 
        DOES NOT contain any header (<th>) tags in its first two rows.
        """
        if not next_file_path: return False
        
        curr_tables = current_soup.find_all('table')
        if not curr_tables: return False
        
        with open(next_file_path, "r", encoding="utf-8") as f:
            next_soup = BeautifulSoup(f.read(), 'html.parser')
        
        next_table = next_soup.find('table')
        if not next_table: return False

        # Check the first few rows of the next table
        next_rows = next_table.find_all('tr')[:2]
        has_header_on_next_page = any(row.find('th') for row in next_rows)

        # If next page table HAS headers, it's a NEW table (don't stitch)
        # If next page table HAS NO headers, it's a CONTINUATION (stitch)
        return not has_header_on_next_page

    def call_llm_for_table_summary(self, table_html):
        prompt = f"""
        You are an expert information extraction assistant.
        Task: Read the following table and produce a complete ASCII/text summary.
              Read the document both vertically and horizontally. 
              Ensure that data points appearing in distinct columns are mapped correctly to their specific entity.
        Rules:
        1. Identify and preserve section titles or headers.
        2. Extract categories, conditions, and values accurately.
        3. Present output in structured ASCII format.

        Constraint: Ensure ZERO information loss. If a number exists in the input, it must appear in your output,
        Input: {table_html}
        Output: Return the ASCII full summary block.
        """
        try:
            response = self.llm.invoke(prompt)
            content = response.content
            self.tracker.add_call(prompt, content)
            return content
        except Exception as e:
            self.logger.error(f"Table summary failed: {str(e)}")
            return f"ERROR_SUMMARIZING: {str(e)}"

    def _summarize_page_if_multi_tables(self, current_soup, page_num: int) -> str:
        tables = current_soup.find_all('table')
        if len(tables) <= 1: return ""
        distinct_tables = []
        prev_headers = None
        for table in tables:
            headers = [th.get_text(strip=True) for th in table.find_all('th')]
            if not headers: headers = [td.get_text(strip=True) for td in table.find_all('td')[:3]]
            if headers != prev_headers: distinct_tables.append(table)
            prev_headers = headers

        if len(distinct_tables) > 1:
            full_text = current_soup.get_text(separator="\n", strip=True)
            prompt = f"""
            You are an expert information extraction assistant.
            Task: Read the following HTML page content and produce a complete ASCII/text summary.
            Rules:
            1. Organize output topic-wise: each topic followed by its summary.
            2. Preserve actual meaning and details (numbers, categories, conditions).
            3. Ensure summaries are long enough to be meaningful.
            4. Format strictly as:
            Topic:<topic_name>
            Summary:
            Topic:<topic_name> 
            Summary:
            ...
            Input (Page {page_num}):
            {full_text}
            Output: Return the ASCII summary block.
            """
            try:
                response = self.llm.invoke(prompt)
                content = response.content
                self.tracker.add_call(prompt, content)
                return content.strip()
            except Exception as e:
                self.logger.error(f"Multi-table page summary failed for page {page_num}: {str(e)}")
        return ""
    



    def _get_overlap(self, next_file_path: Path):
        if not next_file_path: return "", ""
        with open(next_file_path, "r", encoding="utf-8") as f:
            next_soup = BeautifulSoup(f.read(), 'html.parser')
        first_h = next_soup.find(['h1', 'h2', 'h3'])
        stop_header = first_h.get_text().strip() if first_h else ""
        overlap_text = ""
        if first_h:
            prev_nodes = first_h.find_all_previous(string=True)
            overlap_text = " ".join([t.strip() for t in reversed(prev_nodes) if t.strip()])
        else:
            overlap_text = next_soup.get_text()[:300]
        return self._clean_text(overlap_text), stop_header

    def _parse_file(self, file_path: Path, next_file_path: Path, skip_first_table: bool):
            with open(file_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), 'html.parser')

            
            headers = [elem.get_text().strip() for tag in ['h1', 'h2', 'h3', 'h4', 'strong', 'b'] 
                    for elem in soup.find_all(tag) if 10 < len(elem.get_text().strip()) < 150]
            all_text = soup.get_text(separator=" ", strip=True)
            first_1200_words = " ".join(all_text.split()[:1200])

            page_num = int(re.search(r'\d+', file_path.name).group())
            
            # 1. Check for multi-table summary first
            multi_table_summary = self._summarize_page_if_multi_tables(soup, page_num)

            tables_list = []
            did_stitch_forward = False

            # 2. IF MULTI-TABLE SUMMARY EXISTS: Skip individual table processing
            if multi_table_summary:
                self.logger.info(f"Page {page_num}: Multiple tables detected. Summarizing page and skipping table nodes.")
                
                # Clean up the soup for narrative extraction
                for table in soup.find_all('table'):
                    table.decompose()
                
                cleaned_text = self._clean_text(soup.get_text(separator="\n", strip=True))
                full_narrative = f"PAGE_LEVEL_SUMMARY:\n{multi_table_summary}\n\n{cleaned_text}"
                
                with open(self.page_out / f"{file_path.stem}.txt", "w", encoding="utf-8") as pf:
                    pf.write(full_narrative)
                
                # Return [] for tables_list to make it null/empty in JSON
                return full_narrative, [], False, soup, headers, first_1000_words


            # 3. NORMAL PROCESSING: (Only if multi_table_summary is empty)
            all_tables = soup.find_all('table')
            for i, table in enumerate(all_tables):
                if i == 0 and skip_first_table:
                    table.decompose()
                    continue
                
                raw_html = str(table)
                is_split = False
                if i == len(all_tables) - 1 and next_file_path and self._is_table_split(soup, next_file_path):
                    with open(next_file_path, "r", encoding="utf-8") as nf:
                        next_soup = BeautifulSoup(nf.read(), 'html.parser')
                        next_table = next_soup.find('table')
                        if next_table:
                            raw_html = self._stitch_tables(raw_html, str(next_table))
                            is_split = True
                            did_stitch_forward = True

                t_summary = self.call_llm_for_table_summary(raw_html)
                table_id = f"T-{file_path.stem}-{i}"
                tables_list.append({
                    "table_id": table_id, 
                    "table_summary_content": t_summary, 
                    "is_split_across_pages": is_split, 
                    "metadata": None
                })
                
                    # This only runs for single-table pages
                with open(self.table_out / f"{table_id}.md", "w", encoding="utf-8") as tf:
                    tf.write(t_summary)
                table.decompose()

            cleaned_text = self._clean_text(soup.get_text(separator="\n", strip=True))
            with open(self.page_out / f"{file_path.stem}.txt", "w", encoding="utf-8") as pf:
                pf.write(cleaned_text)

            return cleaned_text, tables_list, did_stitch_forward, soup, headers, first_1000_words
    
    def _generate_relevant_title(self, page_num: int, first_1000: str, headers: List[str], tables: List[dict]) -> str:
        table_context = "\n\nTables found:\n" + "\n---\n".join(t.get('table_summary_content', '')[:500] for t in tables[:2]) if tables else ""
        header_context = "\n".join(headers[:8]) if headers else "No clear headers"
        prompt = f"""You are an expert document section title generator.

        Page number: {page_num}

        Generate a short, meaningful section title.

        First 1000 words:
        {first_1000[:1000]}

        Headers:
        {header_context}

        Table summaries:
        {table_context}
        
        Dont overthink
        Also Focus on the table title only if the page contains text and table 
        if different page contains same table name then give the relavent title name from its summery
        If a page contiains multiple table then foucs  on the first paragraph of that page
    
        Return ONLY the title (10-80 characters). Be specific."""
        try:
            response = self.llm.invoke(prompt)
            title = response.content.strip()
            self.tracker.add_call(prompt, title)
            title = re.sub(r'^(Title:|Section:|Page \d+:?)\s*', '', title, flags=re.I).strip('"\'*')
            return title[:87] + "..." if len(title) > 90 else (title if len(title) > 8 else f"Section {page_num}")
        except Exception as e:
            return f"Section {page_num}"

    def _get_first_n_paragraphs(self, soup, n: int = 6) -> str:
        paras = soup.find_all('p')[:n]
        text = " ".join(p.get_text().strip() for p in paras)
        if not text.strip(): text = " ".join(soup.get_text(separator=" ", strip=True).split()[:500])
        return self._clean_text(text)

    def _llm_name_root(self, opening_text: str) -> str:
        prompt = f"""You are an expert document indexer.
        Identify the main document title or plan name from the opening content below.
        It is typically the first prominent heading or title in the first few paragraphs.

Opening content:
{opening_text[:1500]}

Return ONLY the document title (max 80 characters). No preamble, no quotes."""
        try:
            response = self.llm.invoke(prompt)
            title = response.content.strip().strip('"\'*')
            # SYSTEMIZED: Single call tracking
            self.tracker.add_call(prompt, title)
            return title[:80] if len(title) > 5 else "Document Root"
        except Exception as e:
            return "Document Root"

    def run(self):
        if not self.all_files:
            self.logger.warning("No page files found. Exiting.")
            return

        self.logger.info("=" * 30 + " STARTING EXTRACTION PIPELINE " + "=" * 30)
        
        with open(self.all_files[0], "r", encoding="utf-8") as f:
            page1_soup = BeautifulSoup(f.read(), 'html.parser')
        
        root_title = self._llm_name_root(self._get_first_n_paragraphs(page1_soup, n=6))
        root_node = self.factory.create_smart_page(title=root_title, page_num=0, raw_text=" ", parent_id=None)
        real_root_id = root_node["node_id"]

        skip_next_table = False
        child_nodes = []

        for i in range(len(self.all_files)):
            current_file = self.all_files[i]
            next_file = self.all_files[i + 1] if (i + 1) < len(self.all_files) else None
            page_num = int(re.search(r'\d+', current_file.name).group())

            text, tables, was_stitched, soup, headers, first_1000 = self._parse_file(current_file, next_file, skip_next_table)
            title = self._generate_relevant_title(page_num, first_1000, headers, tables)
            overlap_txt, stop_h = self._get_overlap(next_file)

            merged_parts = [f"--- PAGE {page_num} START ---"]
            if text.strip(): merged_parts.append(f"NARRATIVE TEXT:\n{text}")
            if tables:
                merged_parts.append("--- TABLE SUMMARIES ---")
                for t in tables: merged_parts.append(t.get('table_summary_content', ''))
            if overlap_txt.strip(): merged_parts.append(f"CONTEXT OVERLAP:\n{overlap_txt}")
            merged_parts.append(f"--- PAGE {page_num} END ---")

            node = self.factory.create_smart_page(
                title=title, page_num=page_num, raw_text="\n\n".join(merged_parts),
                overlap_text=overlap_txt, stop_header=stop_h, tables=tables, parent_id=real_root_id
            )
            child_nodes.append(node)
            skip_next_table = was_stitched
            self.logger.info(f"Page {page_num:02d}: {title}")

        root_node["merged_content"] = " "
        
        final_nodes = [root_node] + child_nodes
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(final_nodes, f, indent=4, ensure_ascii=False)

        stats = self.tracker.get_report()
        self.logger.info(f"SUCCESS: {len(final_nodes)} nodes created. Tokens: {stats['total_tokens']}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 semantic_page_extractor.py <input_dir> <output_json>")
        sys.exit(1)
    PageExtractor(sys.argv[1], sys.argv[2]).run()

if __name__ == "__main__":
    main()