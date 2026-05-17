# page_schema.py


from typing import List, Optional, TypedDict, Dict, Any

class TableData(TypedDict):
    """
    Represents table data as a summarized string rather than raw rows.
    """
    table_id: str
    table_summary_content: str  # The core facts extracted from the table
    is_split_across_pages: bool
    metadata: Optional[Dict[str, Any]]

class SemanticOverlap(TypedDict):
    """
    Captures the 'leakage' from the next page to preserve sentence meaning.
    """
    text: str
    source_page: Optional[int]
    stopped_at_header: Optional[str] # The title that prevented further overlapping

class PageIndexNode(TypedDict):
    """
    The complete, logically independent unit for each page.
    """
    title: str
    node_id: str
    page_number: int
    has_table: bool
    merged_content: str  # The FINAL string used for RAG (Text + Summary + Overlap)
    tables: List[TableData]
    overlap: SemanticOverlap
    parent_node_id: Optional[str]

# --- The Factory Class ---

class DocumentIndexFactory:
    def __init__(self):
        # Initialize the auto-incrementing ID
        self.current_id = 1

    def _generate_id(self) -> str:
        """Generates a padded 4-digit ID (e.g., '0001')."""
        id_str = str(self.current_id).zfill(4)
        self.current_id += 1
        return id_str

    def create_smart_page(
        self, 
        title: str, 
        page_num: int, 
        raw_text: str,
        overlap_text: str = "",
        stop_header: str = "",
        tables: List[TableData] = None,
        parent_id: Optional[str] = None
    ) -> PageIndexNode:
        """
        Constructs a PageIndexNode. 
        It merges raw text, summarized tables, and semantic overlaps into 
        one single string to ensure the LLM has zero context loss.
        """
        table_list = tables or []
        has_table = len(table_list) > 0
        
        # 1. Format summarized Table Content
        # We wrap the summary in clear tags so the LLM knows it's data
        table_text_blocks = []
        for t in table_list:
            table_text_blocks.append(
                f"\n[DATA TABLE SUMMARY: {t['table_summary_content']}]\n"
            )
        
        table_combined_text = "".join(table_text_blocks)

        # 2. Construct the Merged Content
        # This is the sequence the RAG pipeline will actually 'see'
        merged = (
            f"--- PAGE {page_num} START ---\n"
            f"{raw_text.strip()}\n"
            f"{table_combined_text.strip()}\n"
            f"{overlap_text.strip()}\n"
            f"--- PAGE {page_num} END ---"
        ).strip()

        # 3. Assemble the final node
        return {
            "title": title,
            "node_id": self._generate_id(),
            "page_number": page_num,
            "has_table": has_table,
            "merged_content": merged,
            "tables": table_list,
            "overlap": {
                "text": overlap_text,
                "source_page": page_num + 1 if overlap_text else None,
                "stopped_at_header": stop_header if stop_header else None
            },
            "parent_node_id": parent_id
        }


# # --- Execution Block ---

# if __name__ == "__main__":
#     # 1. Initialize the factory
#     factory = DocumentIndexFactory()

#     # 2. Create a dummy table summary
#     sample_table: TableData = {
#         "table_id": "T-101",
#         "table_summary_content": "Annual Deductible: $500 Individual / $1000 Family. Co-insurance: 20%.",
#         "is_split_across_pages": False,
#         "metadata": None
#     }

#     # 3. Create a smart page node
#     page_node = factory.create_smart_page(
#         title="Medical Benefits Overview",
#         page_num=15,
#         raw_text="The plan provides comprehensive medical coverage for all full-time employees.",
#         overlap_text="Coverage begins on the first day of the month following hire.",
#         stop_header="Dental Benefits",
#         tables=[sample_table],
#         parent_id="ROOT_001"
#     )

#     # 4. Print the output to the terminal
#     import json
#     print("\n--- GENERATED PAGE NODE ---")
#     print(json.dumps(page_node, indent=4))