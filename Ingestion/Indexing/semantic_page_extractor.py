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

        # Hierarchy state
        self.root_node_id = None
        self.last_title = ""   
                       
        # Debug
        self.debug_dir = Path("debug_verification")
        self.page_out = self.debug_dir / "extracted_text"
        self.table_out = self.debug_dir / "table_summaries"
        self.page_out.mkdir(parents=True, exist_ok=True)
        self.table_out.mkdir(parents=True, exist_ok=True)

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

        clean_model = GROQ_MODEL.replace("groq/", "").strip()
        self.llm = ChatGroq(api_key=GROQ_API_KEY, model_name=clean_model, temperature=0)

        self.all_files = self._get_sorted_files()


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
        soup = BeautifulSoup(table_html, 'html.parser')
        rows = soup.find_all('tr')
        return "".join(str(row) for row in rows)

    def _stitch_tables(self, first_table_html: str, second_table_html: str) -> str:
        soup_main = BeautifulSoup(first_table_html, 'html.parser')
        target_table = soup_main.find('table')
        if not target_table:
            return first_table_html
        rows_to_add = self._extract_table_rows(second_table_html)
        new_rows_soup = BeautifulSoup(rows_to_add, 'html.parser')
        container = target_table.find('tbody') if target_table.find('tbody') else target_table
        container.append(new_rows_soup)
        return str(target_table)

    def _is_table_split(self, current_soup, next_file_path):
        if not next_file_path: return False
        if not current_soup.find_all('table'): return False
        with open(next_file_path, "r", encoding="utf-8") as f:
            next_soup = BeautifulSoup(f.read(), 'html.parser')
        return bool(next_soup.find('table'))

    def call_llm_for_summary(self, table_html):
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
            
            # Step 1: Assign the content to a variable so the tracker can use it
            content = response.content
            
            # Step 2: Track the tokens using that variable
            self.tracker.add_call(prompt, content)
            
            # Step 3: Return exactly what you wanted
            return response.content
           
        except Exception as e:
            self.logger.error(f"Table summary failed: {str(e)}")
            return f"ERROR_SUMMARIZING: {str(e)}"

    def _get_page_lead(self, soup):
        all_text = soup.get_text(separator=" ", strip=True)
        words = all_text.split()
        return " ".join(words[:100])

    def _get_overlap_with_indexing_guard(self, next_file_path: Path):
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

        lead_context = self._get_page_lead(soup)

        headers = []
        for tag in ['h1', 'h2', 'h3', 'h4', 'strong', 'b']:
            for elem in soup.find_all(tag):
                text = elem.get_text().strip()
                if 10 < len(text) < 150:
                    headers.append(text)

        all_text = soup.get_text(separator=" ", strip=True)
        first_1000_words = " ".join(all_text.split()[:1000])

        tables_list = []
        all_tables = soup.find_all('table')
        did_stitch_forward = False

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

            t_summary = self.call_llm_for_summary(raw_html)
            table_id = f"T-{file_path.stem}-{i}"
            tables_list.append({
                "table_id": table_id,
                "table_summary_content": t_summary,
                "is_split_across_pages": is_split,
                "metadata": None
            })

            with open(self.table_out / f"{table_id}.md", "w", encoding="utf-8") as tf:
                tf.write(t_summary)

            table.decompose()

        cleaned_text = self._clean_text(soup.get_text(separator="\n", strip=True))

        with open(self.page_out / f"{file_path.stem}.txt", "w", encoding="utf-8") as pf:
            pf.write(cleaned_text)

        return cleaned_text, tables_list, did_stitch_forward, soup, lead_context, headers, first_1000_words


    def _generate_relevant_title(self, page_num: int, first_1000: str, headers: List[str], tables: List[dict]) -> str:
       
        table_context = ""
        if tables:
            table_context = "\n\nTables found:\n" + "\n---\n".join(
                t.get('table_summary_content', '')[:500] for t in tables[:2]
            )
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
        Also Focus on the table title onl tif the page contains text and table 
        if different page contains same table name then give the relavent title name from its summery
        If a page contiains multiple table then foucs  on the first paragraph of that page
    
        Return ONLY the title (10-80 characters). Be specific."""

        try:
            response = self.llm.invoke(prompt)
            raw_content = response.content
            self.tracker.add_call(prompt, raw_content)
            title = response.content.strip()
            title = re.sub(r'^(Title:|Section:|Page \d+:?)\s*', '', title, flags=re.I)
            title = title.strip('"\'*')
            if len(title) > 90:
                title = title[:87] + "..."
            return title if len(title) > 8 else f"Section {page_num}"
        except Exception as e:
            self.logger.error(f"Title generation failed for page {page_num}: {e}")
            return f"Section {page_num}"

    def _get_first_n_paragraphs(self, soup, n: int = 6) -> str:
        """
        Extract text from the first n <p> tags on page 1.
        Used only to identify the ROOT node name.
        Falls back to first 500 words when <p> tags are absent.
        """
        paras = soup.find_all('p')[:n]
        text = " ".join(p.get_text().strip() for p in paras)
        if not text.strip():
            text = " ".join(soup.get_text(separator=" ", strip=True).split()[:500])
        return self._clean_text(text)

    
    def _llm_name_root(self, opening_text: str) -> str:
        """
        Extract the ROOT node name from page 1's opening paragraphs.
        This is a title extraction, not a summary — the summary is built
        separately in _llm_summarize_children() after all pages are processed.
        """
        prompt = f"""You are an expert document indexer.
        Identify the main document title or plan name from the opening content below.
        It is typically the first prominent heading or title in the first few paragraphs.

Opening content:
{opening_text[:1500]}

Return ONLY the document title (max 80 characters). No preamble, no quotes."""
        try:
            response = self.llm.invoke(prompt)
            
            raw_content = response.content
            
            self.tracker.add_call(prompt, raw_content)
            title = self.llm.invoke(prompt).content.strip().strip('"\'*')
            return title[:80] if len(title) > 5 else "Document Root"
        except Exception as e:
            self.logger.error(f"Root title extraction failed: {e}")
            return "Document Root"

    def _llm_summarize_children(self, root_title: str, child_nodes: List[dict]) -> str:
        """
        Generate ROOT node content — a high-level summary of the entire
        document — built from every child node's title + opening text.
        Called AFTER all pages are processed so all children are available.
        """
        children_block = "\n".join(
            f"- [{node['title']}]: "
            f"{' '.join(node.get('raw_text', '').split()[:60])}"
            for node in child_nodes
        )
        prompt = f"""You are an expert document summarizer.
            Document title: "{root_title}"

            Below are all the sections of this document with a short preview of each:

            {children_block}soup

            Write a concise, high-level summary of the entire document (150-250 words).
            Cover what the document is about, what topics it addresses, and who it is
            intended for. Write in clear prose — no bullet points."""
        try:
            
            response = self.llm.invoke(prompt)
            
            # TRACKING LOGIC
            raw_content = response.content
            self.tracker.add_call(prompt, raw_content)
        
            return response.content.strip()

        except Exception as e:
            self.logger.error(f"Root summary generation failed: {e}")
            return "Document covering: " + ", ".join(n["title"] for n in child_nodes)

    def run(self):
        if not self.all_files:
            self.logger.warning("No page files found. Exiting.")
            return

        self.logger.info("=" * 55 + " STARTING EXTRACTION PIPELINE " + "=" * 55)

        # STEP 1: Extract ROOT name from page 1's first 6 paragraphs ────────
        
        with open(self.all_files[0], "r", encoding="utf-8") as f:
            page1_soup = BeautifulSoup(f.read(), 'html.parser')
        
        root_title = self._llm_name_root(self._get_first_n_paragraphs(page1_soup, n=6))
        
        root_node = self.factory.create_smart_page(
            title=root_title,
            page_num=0,
            raw_text="PENDING_SUMMARY", # Placeholder, will update later
            parent_id=None 
        )
        real_root_id = root_node["node_id"]
        
        self.logger.info(f"ROOT NODE  → {root_title} ")


        # STEP 2: Process every page as CHILD ROOT
        skip_next_table = False
        child_nodes = []

        for i in range(len(self.all_files)):
            current_file = self.all_files[i]
            next_file = self.all_files[i + 1] if (i + 1) < len(self.all_files) else None
            page_num = int(re.search(r'\d+', current_file.name).group())

            text, tables, was_stitched, soup, lead_context, headers, first_1000 = self._parse_file(
                current_file, next_file, skip_next_table
            )

            # Every page is a CHILD ROOT
            title = self._generate_relevant_title(page_num, first_1000, headers, tables)

            overlap_txt, stop_h = self._get_overlap_with_indexing_guard(next_file)

            merged_parts = [f"--- PAGE {page_num} START ---"]
            merged_parts.append(f"PAGE IDENTITY:\n{lead_context}")
            if text.strip():
                merged_parts.append(f"NARRATIVE TEXT:\n{text}")
            if tables:
                merged_parts.append("--- TABLE SUMMARIES ---")
                for t in tables:
                    merged_parts.append(t.get('table_summary_content', ''))
            if overlap_txt.strip():
                merged_parts.append(f"CONTEXT OVERLAP:\n{overlap_txt}")
            merged_parts.append(f"--- PAGE {page_num} END ---")
            full_merged_text = "\n\n".join(merged_parts)

            node = self.factory.create_smart_page(
                title=title,
                page_num=page_num,
                raw_text=full_merged_text,
                overlap_text=overlap_txt,
                stop_header=stop_h,
                tables=tables,
                parent_id=real_root_id,       #
            )

            self.logger.info(f"Page {page_num:02d}: CHILD ROOT → {title}")
            child_nodes.append(node)
            skip_next_table = was_stitched

        # ── STEP 3: Create ROOT node with summary of ALL children ──────────────
        self.logger.info("Building ROOT node summary from all child nodes ...")
        root_summary = self._llm_summarize_children(root_title, child_nodes)
        root_node["merged_content"] = root_summary
    

        # ── STEP 4: Write — root first, then children in page order ───────────
        final_nodes = [root_node] + child_nodes
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(final_nodes, f, indent=4, ensure_ascii=False)

        self.logger.info(
            f"SUCCESS: JSON created at {self.output_path} "
            f"| 1 ROOT + {len(child_nodes)} CHILD ROOT nodes "
            f"| Total: {len(final_nodes)}"
        )
        # Final reporting
        stats = self.tracker.get_report()
        self.logger.info("=" * 30)
        self.logger.info(f"FINAL TOKEN USAGE REPORT")
        self.logger.info(f"Total LLM Calls: {stats['total_calls']}")
        self.logger.info(f"Input Tokens:  {stats['input_tokens']}")
        self.logger.info(f"Output Tokens: {stats['output_tokens']}")
        self.logger.info(f"Grand Total:   {stats['total_tokens']}")
        self.logger.info("=" * 30)
        
        



def main():
    if len(sys.argv) < 3:
        print("Usage: python3 semantic_page_extractor.py <input_dir> <output_json>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_file = sys.argv[2]

    extractor = PageExtractor(input_dir, output_file)
    extractor.run()


if __name__ == "__main__":
    main()