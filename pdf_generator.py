from fpdf import FPDF
from PIL import Image
import io
import requests
from datetime import datetime
import os
import tempfile
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def clean_text(text):
    """Clean text by replacing problematic characters with their ASCII equivalents"""
    if not isinstance(text, str):
        return str(text)
    
    # Replace smart quotes and other special characters
    replacements = {
        '"': '"',  # smart quote
        '"': '"',  # smart quote
        ''': "'",  # smart apostrophe
        ''': "'",  # smart apostrophe
        '–': '-',  # en dash
        '—': '-',  # em dash
        '…': '...',  # ellipsis
        '\u2022': '*',  # bullet point
        '\u2013': '-',  # en dash
        '\u2014': '-',  # em dash
        '\u2018': "'",  # left single quote
        '\u2019': "'",  # right single quote
        '\u201C': '"',  # left double quote
        '\u201D': '"',  # right double quote
        '\u2026': '...',  # horizontal ellipsis
        '\u2028': ' ',  # line separator
        '\u2029': ' ',  # paragraph separator
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove any remaining non-ASCII characters
    text = text.encode('ascii', 'replace').decode('ascii')
    return text

class CaptionPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Caption Review Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_caption_pdf(captions_data, output_path):
    logger.info("Starting PDF generation...")
    pdf = CaptionPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    temp_files = []
    
    try:
        for idx, row in enumerate(captions_data):
            logger.info(f"Processing row {idx + 1}/{len(captions_data)}")
            pdf.add_page('P')
            
            # Title
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, 'Image Captions Review', 0, 1, 'C')
            pdf.ln(5)
            
            # Product Image and Captions Section
            start_y = pdf.get_y()
            
            # Product Image
            try:
                logger.debug(f"Fetching product image from URL: {row['product_image_url']}")
                response = requests.get(row['product_image_url'])
                img = Image.open(io.BytesIO(response.content))
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    img_path = tmp.name
                    temp_files.append(img_path)
                    img.save(img_path)
                    # Place image on the left side
                    pdf.image(img_path, x=10, y=start_y, h=60)
            except Exception as e:
                logger.error(f"Error loading product image: {str(e)}")
                pdf.set_xy(10, start_y)
                pdf.cell(0, 10, f"Error loading product image: {str(e)}", 0, 1, 'L')
            
            # Product Captions (on the right side)
            pdf.set_xy(90, start_y)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Product Image Captions:', 0, 1)
            
            pdf.set_font('Arial', 'B', 10)
            pdf.set_x(90)
            pdf.cell(0, 10, 'Short Caption:', 0, 1)
            
            pdf.set_font('Arial', '', 10)
            pdf.set_x(90)
            pdf.multi_cell(0, 5, clean_text(row['short_caption']), 0, 'L')
            
            pdf.ln(5)
            pdf.set_font('Arial', 'B', 10)
            pdf.set_x(90)
            pdf.cell(0, 10, 'Long Caption:', 0, 1)
            
            pdf.set_font('Arial', '', 10)
            pdf.set_x(90)
            pdf.multi_cell(0, 5, clean_text(row['long_caption']), 0, 'L')
            
            pdf.ln(10)
            
            # Lifestyle Image and Caption Section
            if row['lifestyle_image_url'] and pd.notna(row['lifestyle_image_url']):
                start_y = pdf.get_y()
                
                try:
                    logger.debug(f"Fetching lifestyle image from URL: {row['lifestyle_image_url']}")
                    response = requests.get(row['lifestyle_image_url'])
                    img = Image.open(io.BytesIO(response.content))
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        img_path = tmp.name
                        temp_files.append(img_path)
                        img.save(img_path)
                        # Place image on the left side
                        pdf.image(img_path, x=10, y=start_y, h=60)
                except Exception as e:
                    logger.error(f"Error loading lifestyle image: {str(e)}")
                    pdf.set_xy(10, start_y)
                    pdf.cell(0, 10, f"Error loading lifestyle image: {str(e)}", 0, 1, 'L')
                
                # Lifestyle Caption (on the right side)
                pdf.set_xy(90, start_y)
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 10, 'Lifestyle Image Caption:', 0, 1)
                
                pdf.set_font('Arial', '', 10)
                pdf.set_x(90)
                pdf.multi_cell(0, 5, clean_text(row['lifestyle_caption']), 0, 'L')
            
            pdf.ln(10)
        
        # Save the PDF
        logger.info("Saving PDF...")
        if isinstance(output_path, io.BytesIO):
            logger.debug("Output is BytesIO object")
            # If output_path is a BytesIO object, write to a temporary file first
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                temp_path = tmp.name
                temp_files.append(temp_path)
                logger.debug(f"Created temporary PDF file: {temp_path}")
                pdf.output(temp_path)
                # Read the temporary file and write to BytesIO
                with open(temp_path, 'rb') as f:
                    output_path.write(f.read())
                output_path.seek(0)
        else:
            logger.debug(f"Output is file path: {output_path}")
            # If output_path is a file path, save directly
            pdf.output(output_path)
        logger.info("PDF generation completed successfully")
            
    except Exception as e:
        logger.error(f"Error during PDF generation: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up temporary files
        logger.debug(f"Cleaning up {len(temp_files)} temporary files")
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
                logger.debug(f"Removed temporary file: {temp_file}")
            except Exception as e:
                logger.error(f"Error removing temporary file {temp_file}: {str(e)}")
                pass
