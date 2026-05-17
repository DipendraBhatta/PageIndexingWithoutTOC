import json
import logging
import sys
from pathlib import Path
from typing import List

from ingestion.indexing.page_schema import PageIndexNode
from ingestion.indexing.schema import IndexNode


class IndexingAggregator:
    """
    Aggregates flat PageIndexNodes into hierarchical IndexNode structure.
    Automatically calculates start_index and end_index based on page numbers.
    """

    def __init__(self, input_path, output_path):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.logger = logging.getLogger("IndexingLogger")



    def build(self, pages: List[PageIndexNode]) -> IndexNode:
        """
        Builds hierarchical index from flat pages based on parent_node_id.
        """
        if not pages:
            self.logger.error("No pages provided.")
            raise ValueError("Pages list cannot be empty.")

        # Step 1: Identify the Root Node
        # Logic: parent_node_id is None AND page_number is 0
        root_page = next((p for p in pages if p.get("parent_node_id") is None and p.get("page_number") == 0), None)

        if not root_page:
            self.logger.error("Root node not found.")
            raise ValueError("Root node not found (page_number=0 and parent_node_id=None)")

        # Step 2: Identify Child Nodes
        # Logic: Nodes whose parent_node_id matches the root_page's node_id
        root_id = root_page["node_id"]
        child_pages = [p for p in pages if p.get("parent_node_id") == root_id]
        
        # Sort children by page number to ensure order
        child_pages.sort(key=lambda p: p["page_number"])

        self.logger.info(f"Building hierarchy for Root ID: {root_id} | Children found: {len(child_pages)}")

        # Step 3: Build child nodes
        child_nodes = []
        for page in child_pages:
            child_node: IndexNode = {
                "title": page["title"],
                "node_id": page["node_id"],
                "start_index": page["page_number"],
                "end_index": page["page_number"], 
                "summary": None,
                "content": page["merged_content"],
                "nodes": []
            }
            child_nodes.append(child_node)

        # Step 4: Build Root Node
        # start_index = first child page number end_index = last child's page number
        root_start = child_pages[0]["page_number"]
        root_end = child_pages[-1]["page_number"] if child_pages else 0

        root_node: IndexNode = {
            "title": root_page["title"],
            "node_id": root_page["node_id"],
            "start_index": root_start, 
            "end_index": root_end,
            "summary": root_page["merged_content"],
            "content": None,
            "nodes": child_nodes
        }

        return root_node

    def save(self, index: IndexNode, output_path: Path) -> None:
        """Save the final hierarchical index to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=4, ensure_ascii=False)

        self.logger.info(f"Final hierarchical index saved successfully at: {output_path}")

    def run(self):
        """Main run method"""
        self.logger.info("=" * 65 + " STARTING HIERARCHICAL AGGREGATION " + "=" * 65)

        try:
            with open(self.input_path, "r", encoding="utf-8") as f:
                flat_pages: List[PageIndexNode] = json.load(f)

            self.logger.info(f"Loaded {len(flat_pages)} flat nodes from {self.input_path}")

            hierarchical_index = self.build(flat_pages)
            self.save(hierarchical_index, self.output_path)

            self.logger.info(" Hierarchical indexing completed successfully!")
        except Exception as e:
            self.logger.error(f"Error during aggregation: {str(e)}", exc_info=True)


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 indexing_aggregator.py <flat_input_json> <hierarchical_output_json>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    aggregator = IndexingAggregator(input_file, output_file)
    aggregator.run()


if __name__ == "__main__":
    main()