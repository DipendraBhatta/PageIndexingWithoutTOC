# DOCUMENT INDEXING AND RETRIEVAL SYSTEM (RAG)

This project provides a complete pipeline for parsing PDF documents, extracting semantic structures, indexing data into a vector store (ChromaDB), and retrieving information through a query engine.

## 1. ENVIRONMENT AND UTILITIES
The project uses a .env file for API credentials. Environment management and shared utility functions are handled by utils.py. Ensure your virtual environment is active before running commands.

## 2. INGESTION PHASE: PARSING
Navigate to the root directory and execute the parsing scripts. These scripts convert raw PDFs into manageable data structures.

### Step A: Core Parsing
Run the main parser to extract initial data:
python3 -m Ingestion.Parse.parse_main

### Step B: HTML Structuring
Convert the parsed data into structured HTML for better semantic clarity:
python3 -m Ingestion.Parse.structured_html

## 3. INGESTION PHASE: INDEXING
After parsing, move to the indexing stage to create a searchable database.

### Step A: Semantic Extraction
Extract meaningful page-level information:
python3 -m Ingestion.Indexing.semantic_page_extractor

### Step B: Index Aggregation
Aggregate all extracted data into the final indexing format:
python3 -m Ingestion.Indexing.Indexing_aggregator

This will generate the 'Final_Indexing.json' file located in 'Ingestion_Results/Indexing_Results/'.

## 4. QUERY AND RETRIEVAL PHASE
Once indexing is complete, you can interact with the data through the retrieval engine.

### Run Retrieval Engine
Execute the main retrieval script to start querying your indexed documents:
python3 -m Query_Retrieval.Retrieval_main

## PROJECT DIRECTORY STRUCTURE
- Ingestion/
  - Parse/           : PDF to Text/HTML conversion scripts.
  - Indexing/        : Semantic extraction and ChromaDB indexing.
- Query_Retrieval/   : Engine for searching and retrieving context.
- Ingestion_Results/ : Stores parsing outputs and final JSON indices.
- utils.py           : API and environment configuration helper.
- .env               : Local environment variables (API keys).

## LOGS AND DEBUGGING
Process logs are stored in 'indexing_process.log' and 'process.log'. 
Detailed parsing logs can be found in the 'DoclingParsing_Log/' directory.
