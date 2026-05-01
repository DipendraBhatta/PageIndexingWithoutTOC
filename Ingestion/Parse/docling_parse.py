import logging
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, 
    AcceleratorOptions, 
    AcceleratorDevice,
    TableFormerMode # Added for cleaner reference
)

class DoclingEngine:
 
    def __init__(self, use_gpu: bool = True):
        options = PdfPipelineOptions()
        
        # --- STABLE PARAMETERS (Matches your working script) ---
        options.do_ocr = True
        options.do_table_structure = True
        options.table_structure_options.do_cell_matching = True
        
        # --- CRITICAL CHANGES FOR ORDER & FLOW ---
        
        # 1. COMMENTED OUT: This was the line causing the "ValueError" and crash.
        # options.layout_options.do_postprocessing = True 
        
        # 2. KEPT: This is the actual fix for the side-by-side calendars.
        # By upscaling, you make the gaps between Oct, Nov, and Dec clear to the AI.
        options.images_scale = 2.5 

        # 3. KEPT: Ensures the model uses the high-accuracy engine for table grids.
        options.table_structure_options.mode = TableFormerMode.ACCURATE
        # ------------------------------------------

        # Setup Accelerator (Handles the CPU fallback you saw in your terminal)
        device = AcceleratorDevice.AUTO if use_gpu else AcceleratorDevice.CPU
        options.accelerator_options = AcceleratorOptions(
            num_threads=4, 
            device=device
        )

        self.converter = DocumentConverter(
            format_options={
                # NOTICE: Wrapped in PdfFormatOption to match your working script exactly
                InputFormat.PDF: PdfFormatOption(pipeline_options=options)
            }
        )

    def convert_file(self, file_path):
        """
        Step 1: The 'Master' Conversion.
        """
        result = self.converter.convert(file_path)
        return result.document

    # --- FORMAT SPECIFIC EXPORTERS ---

    def get_markdown(self, doc, page_no=None): # Added default None
        return doc.export_to_markdown(page_no=page_no)

    def get_html(self, doc, page_no=None): # Added default None
        return doc.export_to_html(page_no=page_no)

    def get_json_data(self, doc):
        return doc.export_to_dict()

    # # ADDED: Placeholder to ensure main.py doesn't throw an AttributeError
    # def get_structured_hybrid_page(self, doc, page_no):
    #     """Current task: redirects to standard HTML export"""
    #     return self.get_html(doc, page_no)