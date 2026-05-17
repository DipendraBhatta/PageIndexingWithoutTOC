import logging
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, 
    AcceleratorOptions, 
    AcceleratorDevice,
    TableFormerMode
)

class DoclingEngine:
 
    def __init__(self, use_gpu: bool = True):
        # 1. Initialize the pipeline options specific to PDF processing
        options = PdfPipelineOptions()
        
        # 2. Enable Optical Character Recognition (OCR) for scanned text detection
        options.do_ocr = True
        
        # 3. Enable the detection and structural analysis of tables within the document
        options.do_table_structure = True
        
        # 4. Enable cell-level matching to align text accurately within table grids
        options.table_structure_options.do_cell_matching = True
        
        # 5. Increase image scale to 2.5x to improve AI vision on complex or small layouts
        options.images_scale = 2.5 

        # 6. Set the table extraction mode to 'ACCURATE' for high-precision grid detection
        options.table_structure_options.mode = TableFormerMode.ACCURATE

        # 7. Determine the hardware device (GPU or CPU) based on system availability
        device = AcceleratorDevice.AUTO if use_gpu else AcceleratorDevice.CPU
        
        # 8. Configure multi-threading and hardware acceleration for faster processing
        options.accelerator_options = AcceleratorOptions(
            num_threads=4, 
            device=device
        )

        # 9. Instantiate the DocumentConverter with the custom PDF configuration
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=options)
            }
        )

    def convert_file(self, file_path):
        # 10. Execute the core conversion process to transform the PDF into a document object
        result = self.converter.convert(file_path)
        
        # 11. Return the extracted document structure for further processing
        return result.document

    def get_markdown(self, doc, page_no=None):
        # 12. Export the document or a specific page into clean Markdown format
        return doc.export_to_markdown(page_no=page_no)

    def get_html(self, doc, page_no=None):
        # 13. Export the document or a specific page into structured HTML format
        return doc.export_to_html(page_no=page_no)

    def get_json_data(self, doc):
        # 14. Export the entire document structure into a Python dictionary for JSON storage
        return doc.export_to_dict()