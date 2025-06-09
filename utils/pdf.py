import logging

import PyPDF2
from pathlib import Path

logger = logging.getLogger(__name__)

def extract_pdf_content(file_path: Path) -> str:
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            content = ""
            
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                except Exception as e:
                    logger.warning(f"Could not extract text from page {page_num + 1}: {e}")
                    
            return content.strip()
            
    except Exception as e:
        logger.error(f"Error extracting PDF content from {file_path}: {e}")
        raise