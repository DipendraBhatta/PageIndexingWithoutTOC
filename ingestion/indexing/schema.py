# schema.py

from typing import List, Optional, TypedDict

# This defines the structure (the "shape") of each entry in your index
class IndexNode(TypedDict):
    title: str
    node_id: str
    start_index: int          # Starting Page Number
    end_index: Optional[int]     # Ending Page Number
    summary: Optional[str]       # LLM generated summary
    content: Optional[str]       # The actual text content from those pages
    nodes: List['IndexNode']     # Recursive children (nested sections)
    

# This class is the "Factory" that creates your nodes and manages IDs
class DocumentIndex:
    def __init__(self):
        # We start with an integer so we can perform math (incrementing)
        self.current_id = 1

    def _generate_id(self) -> str:
        """
        Generates a padded ID string like '0001', '0002', etc.
        """
        # zfill(4) ensures the string is 4 digits long with leading zeros
        id_str = str(self.current_id).zfill(4)
        self.current_id += 1
        return id_str

    def create_node(self, title: str, page: int, summary: str = None, content: str = None) -> IndexNode:
        """
        Helper method to create a dictionary that matches the IndexNode schema.
        """
        
        return {
            "title": title,
            "node_id": self._generate_id(),
            "start_index":page,
            "end_index": None, # This will be filled by the finalization logic later
            "summary": summary,
            "content": content,
            "nodes": []
        }