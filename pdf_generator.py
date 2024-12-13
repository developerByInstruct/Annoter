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
        self.cell(0, 10, 'Model Caption Comparison', 0, 1, 'C')
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
            pdf.add_page('L')  # Landscape orientation for better column layout
            
            # Title
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, 'Comparison of Models', 0, 1, 'C')
            pdf.ln(5)
            
            # Images Section - Side by side
            start_y = pdf.get_y()
            logger.debug(f"Starting Y position for images: {start_y}")
            
            # Product Image
            try:
                logger.debug(f"Fetching product image from URL: {row['product_image_url']}")
                response = requests.get(row['product_image_url'])
                img = Image.open(io.BytesIO(response.content))
                logger.debug(f"Product image size: {img.size}")
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    img_path = tmp.name
                    temp_files.append(img_path)
                    img.save(img_path)
                    logger.debug(f"Saved product image to temp file: {img_path}")
                    pdf.image(img_path, x=10, y=start_y, h=40)
            except Exception as e:
                logger.error(f"Error loading product image: {str(e)}")
                pdf.set_xy(10, start_y)
                pdf.cell(0, 10, f"Error loading product image: {str(e)}", 0, 1, 'L')
            
            # Lifestyle Image
            try:
                if row['lifestyle_image_url'] and pd.notna(row['lifestyle_image_url']):
                    logger.debug(f"Fetching lifestyle image from URL: {row['lifestyle_image_url']}")
                    response = requests.get(row['lifestyle_image_url'])
                    img = Image.open(io.BytesIO(response.content))
                    logger.debug(f"Lifestyle image size: {img.size}")
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        img_path = tmp.name
                        temp_files.append(img_path)
                        img.save(img_path)
                        logger.debug(f"Saved lifestyle image to temp file: {img_path}")
                        pdf.image(img_path, x=150, y=start_y, h=40)
                else:
                    logger.debug("No lifestyle image URL provided or URL is NaN")
            except Exception as e:
                logger.error(f"Error loading lifestyle image: {str(e)}")
                pdf.set_xy(150, start_y)
                pdf.cell(0, 10, f"Error loading lifestyle image: {str(e)}", 0, 1, 'L')
            
            pdf.ln(45)  # Space after images
            
            # Model Captions Table Headers
            models = ['Gemini', 'OpenAI', 'Together', 'Groq', 'Grok']
            col_width = 55
            row_height = 5  # Reduced row height
            padding = 2  # Add padding
            
            # Headers
            pdf.set_font('Arial', 'B', 10)
            start_x = pdf.get_x()
            logger.debug(f"Starting X position for captions: {start_x}")
            for i, model in enumerate(models):
                x = start_x + (i * col_width)
                pdf.set_xy(x, pdf.get_y())
                pdf.cell(col_width, row_height*2, model, 1, 0, 'C')
            pdf.ln(row_height*2)
            
            # Set smaller font size for captions
            pdf.set_font('Arial', '', 7)
            
            # Function to handle caption section
            def draw_caption_section(caption_type, start_x):
                logger.debug(f"Drawing {caption_type} section")
                
                # Calculate total height needed before drawing
                max_height = 0
                captions = []
                
                # First pass: calculate maximum height and store captions
                for model in models:
                    # Increase text limit and handle None/NaN values
                    caption_text = row.get(f'{model.lower()}_{caption_type}', '')
                    if pd.isna(caption_text):
                        caption_text = ''
                    caption = clean_text(str(caption_text))[:800]  # Clean text and limit length
                    captions.append(caption)
                    
                    # Create a temporary PDF object to measure height
                    temp_pdf = FPDF()
                    temp_pdf.set_font('Arial', '', 7)
                    temp_pdf.add_page()
                    temp_pdf.set_xy(0, 0)
                    temp_pdf.multi_cell(w=col_width-2*padding, h=row_height, txt=caption, border=0, align='L')
                    height = temp_pdf.get_y()
                    max_height = max(max_height, height)
                    logger.debug(f"Caption height for {model}: {height}")
                
                # Add padding
                max_height += 2*padding
                
                # Check if we need a page break for long sections
                if caption_type in ['long_caption', 'lifestyle_caption']:
                    max_height = max(max_height, 100)  # Minimum height for long sections
                    if pdf.get_y() + max_height > pdf.h - 20:  # 20 is margin from bottom
                        logger.debug(f"Adding page break before {caption_type}")
                        pdf.add_page('L')  # Add new landscape page
                        # Redraw headers on new page
                        pdf.set_font('Arial', 'B', 10)
                        for i, model in enumerate(models):
                            x = start_x + (i * col_width)
                            pdf.set_xy(x, pdf.get_y())
                            pdf.cell(col_width, row_height*2, model, 1, 0, 'C')
                        pdf.ln(row_height*2)
                        pdf.set_font('Arial', '', 7)
                
                y_start = pdf.get_y()
                logger.debug(f"Drawing section at Y: {y_start}, Max height: {max_height}")
                
                # Second pass: draw cells with uniform height
                for i, caption in enumerate(captions):
                    x = start_x + (i * col_width)
                    # Draw border
                    pdf.rect(x, y_start, col_width, max_height)
                    # Draw text with word wrap
                    pdf.set_xy(x + padding, y_start + padding)
                    pdf.multi_cell(w=col_width-2*padding, h=row_height, txt=caption, border=0, align='L')
                    # Reset Y position for next column
                    pdf.set_y(y_start)
                
                return y_start + max_height
            
            # Draw short captions
            pdf.set_font('Arial', 'B', 9)
            pdf.cell(0, 10, 'Short Captions:', 0, 1, 'L')
            pdf.set_font('Arial', '', 7)
            next_y = draw_caption_section('short_caption', start_x)
            pdf.set_y(next_y + 5)
            logger.debug(f"Short captions section ends at Y: {next_y}")
            
            # Draw long captions
            pdf.set_font('Arial', 'B', 9)
            pdf.cell(0, 10, 'Long Captions:', 0, 1, 'L')
            pdf.set_font('Arial', '', 7)
            next_y = draw_caption_section('long_caption', start_x)
            pdf.set_y(next_y + 5)
            logger.debug(f"Long captions section ends at Y: {next_y}")

            # Draw lifestyle images (moved after long captions)
            if any(row.get(f'{model.lower()}_lifestyle_image') for model in models):
                pdf.set_font('Arial', 'B', 9)
                pdf.cell(0, 10, 'Lifestyle Images:', 0, 1, 'L')
                y_start = pdf.get_y()
                
                for i, model in enumerate(models):
                    x = start_x + (i * col_width)
                    lifestyle_url = row.get(f'{model.lower()}_lifestyle_image')
                    if lifestyle_url and not pd.isna(lifestyle_url):
                        try:
                            img_data = requests.get(lifestyle_url).content
                            if img_data:
                                img = Image.open(io.BytesIO(img_data))
                                # Calculate image dimensions to fit column
                                img_width = col_width - 2*padding
                                ratio = img_width / float(img.size[0])
                                img_height = float(img.size[1]) * ratio
                                
                                # Save image to temp file
                                temp_img = io.BytesIO()
                                img.save(temp_img, format='PNG')
                                temp_img.seek(0)
                                
                                # Draw image
                                pdf.image(temp_img, x=x+padding, y=y_start, w=img_width)
                                next_y = max(next_y, y_start + img_height)
                        except Exception as e:
                            logger.error(f"Error processing lifestyle image for {model}: {e}")
                            pdf.set_xy(x+padding, y_start)
                            pdf.multi_cell(w=col_width-2*padding, h=row_height, 
                                         txt="Error loading lifestyle image", border=0)
                
                pdf.set_y(next_y + 5)
                logger.debug(f"Lifestyle images section ends at Y: {next_y}")
            
            # Draw lifestyle captions
            if any(row.get(f'{model.lower()}_lifestyle_caption') for model in models):
                pdf.set_font('Arial', 'B', 9)
                pdf.cell(0, 10, 'Lifestyle Captions:', 0, 1, 'L')
                pdf.set_font('Arial', '', 7)
                next_y = draw_caption_section('lifestyle_caption', start_x)
                pdf.set_y(next_y + 5)
                logger.debug(f"Lifestyle captions section ends at Y: {next_y}")
            
            pdf.ln(10)  # Space between products
            logger.debug("Finished processing row")
        
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
