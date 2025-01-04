import streamlit as st
import pandas as pd
import google.generativeai as genai
from PIL import Image
import requests
import json
import os
from openai import OpenAI
from together import Together
from groq import Groq
import io
from config import MODELS, OPENAI_API_KEY, TOGETHER_API_KEY, GROQ_API_KEY, GOOGLE_API_KEY, GROK_API_KEY
from pdf_generator import create_caption_pdf
import base64
import zipfile
from scraper_page import display_scraper_page
from data_pipeline import DataPreparationPipeline

# Initialize API clients
genai.configure(api_key=GOOGLE_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
together_client = Together(api_key=TOGETHER_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)
xai_client = OpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1")

sys_prompt = """You are a professional objective product copywriter. You are not subjective, nor do you speak in a marketing tone. Your task is to:  
- Provide the short caption and long caption for a **product image**.  
- Provide only a long caption for a **lifestyle image**.  
- If a **lifestyle image** is supplied in place of a **product image**, describe only the product in the image and ignore all non-product elements, such as the setting or human interactions.

You will be provided with a product link that helps identify the specific product. Use this information to:
1. Accurately identify and describe the product in both product and lifestyle images
2. Ensure consistency between the product name/type in the link and your descriptions
3. Help distinguish the product when it appears in lifestyle images, highlighting the product in the context of its use in lifestyle

### What is the product image?  
The product image should:  
- Be isolated.  
- Feature either a transparent or solid color background.  
- Use a neutral or plain background.  

#### Objects in the image:  
- The image never includes humans. Objects in an image could come in pairs/triplets/etc.  
- Objects may not always be located at the focal point of the image.  
- The image should not include text or graphics unless they are printed on the product itself.  
- The image must not contain artefacts such as stripes, compression artefacts, marks, or highlights.  
- The aspect ratio should not be extreme (>4).  
- The product should not be depicted in a real-life environment or setting. The image must focus on showing the product in high detail from a specific angle or series of angles (if multiple product images are provided).  
- The entire product should be visible.  

#### Packaging:  
- If the product has packaging, focus on the product itself, not the packaging.  

#### Lighting and Colors Requirements:  
- The lighting must not distort or hide the original product details by being too dark or too bright.  
- The color of the object must be accurate.  

#### To add captions:  

**Short Description:**  
- Emphasize simplicity.  
- Focus only on key elements such as what the object is, its function, material, and color.  
- Limit to 5-15 words, with a maximum of 20-25 words.  
- Avoid excessive details and focus on the main attributes.  

Example:  
Short caption: A cordless vacuum with a light blue base and silver and gold handle.  

**Long Description:**  
- Include all specific details visible in the image, such as color, material, features, and brand.  
- Do not include information that can only be learned from external sources (e.g., product description).  
- Positioning should be described based on your perspective (e.g., left, right).  

Example:  
Long caption: A white round jar with a blue cover. The jar has a blue lotus flower design with a vertical blue line that runs down the packaging. The text "DERMA - E" is written below in dark color. Below that is "Eczema Relief Cream" in black font color. The size is written as 4 OZ/ 113g at the lower right corner of the jar.  

### What is a lifestyle image?  
A lifestyle image depicts the product in use within a real-life setting, showcasing its practical application and context.  

#### Handling lifestyle images supplied in place of product images:  
- When a lifestyle image is provided **instead** of a product image, focus **exclusively** on describing the product.  
- Ignore the setting, human interactions, and other elements unrelated to the product.  
- Treat the product as if it were isolated, and provide a detailed description of its visible attributes (e.g., color, material, textures).  

#### Objects in the image:  
- The image may feature humans, multiple instances of the product, or a relatable environment.  
- The product may not always be at the focal point.  
- Text or graphics should not appear unless printed on the product itself.  
- Artefacts such as stripes, compression artefacts, marks, or highlights must be avoided.  
- The aspect ratio should not be extreme (>4).  
- The whole product must be visible.  

#### Lighting and Colors Requirements:  
- The lighting must not distort or hide the original product details by being too dark or too bright.  
- The color of the object must be accurate.  

#### To add captions:  
- Focus on describing the main product and its details when a lifestyle image is supplied in place of a product image. Ignore the background and human interactions unless they are essential for identifying the product.  
- Include details about the product’s position, shapes, colors, textures, or patterns.  
- Avoid subjective descriptions (e.g., “adding a touch of life”).  
- Ensure the description is sufficient for recreating the image.  

Examples:

**Lifestyle image provided in place of a product image**:  
Long caption: The image shows a round white jar with a teal lid and text printed on the front. The text reads "DERMA E" in a bold sans-serif font, followed by the product name. The jar is centered in the frame, with no visible setting or context described.

**Regular lifestyle image**:  
Long caption: The image shows a woman with shoulder-length blonde hair holding a jar of skin product. The woman is light-skinned and is smiling broadly, showcasing her teeth. She is wearing a purple t-shirt. The jar she holds is white with teal-colored accents and text. The text on the jar appears to be the brand name "DERMA E" in a sans-serif, capital font. The jar is centered in her palm, and her hand is holding it. The background is a plain, off-white wall.  

Remember:  
1. Always be objective, and do not use subjective statements like: possibly, maybe, likely, appears to.  
2. Only caption what you see, and do not infer details.  
3. Always begin with 'The image...'"""

def download_image(url):
    try:
        if pd.isna(url):
            return None
        response = requests.get(url)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
        return img
    except Exception as e:
        st.error(f"Error downloading image: {str(e)}")
        return None

def generate_prompt(image_type="product", product_link=None):
    base_prompt = """
Analyze this {type} image and provide captions in the following JSON format. 
Product Link: {link}
{format}"""

    if image_type == "product":
        criteria = """- Focus on physical attributes: dimensions, materials, colors, textures
- Use the product link text to help identify and describe the product accurately
- Describe key functional elements and design features
- Note distinctive visual elements and patterns
- Mention any visible branding or product identifiers
- Describe the overall shape and structure
- Include details about finish and surface appearance"""
        
        json_format = """{
    "short_caption": "5-15 words focusing on key elements (object type, function, material, color). Always begin short caption with 'A...'",
    "long_caption": "Detailed description of all visual elements, focusing on physical attributes visible in the image. Include materials, colors, textures, and design features. Always begin long caption with 'The image...'"
}"""
    else:  # lifestyle
        criteria = """- Describe the product's placement and context
- Note any human interaction or positioning
- Detail the environmental setting
- Describe lighting and atmosphere
- Include relevant background elements
- Focus on how the product is being used or displayed
- Always begin with 'The image...'"""
        
        json_format = """{
    "lifestyle_caption": "Detailed description focusing on product placement, environment, and context. Include relevant details about setting and any human elements present."
}"""

    return base_prompt.format(type=image_type, link=product_link or "Not provided", format=json_format)

def extract_json_from_text(text):
    if not text:
        return None
        
    text = text.strip()
    
    # Try to find JSON block
    json_start = text.rfind('```json')
    json_end = text.rfind('```')
    
    if json_start != -1 and json_end != -1 and json_start < json_end:
        # Extract the JSON content
        json_text = text[json_start + 7:json_end].strip()
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            pass
    
    # If no JSON block found, try to extract from plain text
    try:
        # Look for key phrases that indicate the start of captions
        short_start = text.find("short_caption")
        long_start = text.find("long_caption")
        
        if short_start != -1 or long_start != -1:
            # Extract the first clear short and long caption mentions
            short_caption = ""
            long_caption = ""
            
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if "short_caption" in line and not short_caption:
                    short_caption = line.split(":", 1)[1].strip().strip('"').strip("'")
                if "long_caption" in line and not long_caption:
                    long_caption = line.split(":", 1)[1].strip().strip('"').strip("'")
            
            return {
                "short_caption": short_caption or text[:100],
                "long_caption": long_caption or text
            }
    except Exception:
        pass
        
    # If all else fails, create a default structure
    return {
        "short_caption": text[:100],
        "long_caption": text
    }

def generate_captions_with_model(model_name, product_image_url, lifestyle_image_url=None, product_link=None):
    product_prompt = generate_prompt("product", product_link)
    lifestyle_prompt = generate_prompt("lifestyle", product_link)
    
    try:
        # Download and process images
        product_image = download_image(product_image_url)
        if product_image is None:
            raise ValueError(f"Failed to download product image from {product_image_url}")
            
        lifestyle_image = None
        if lifestyle_image_url and not pd.isna(lifestyle_image_url):
            lifestyle_image = download_image(lifestyle_image_url)
            if lifestyle_image is None:
                st.warning(f"Failed to download lifestyle image from {lifestyle_image_url}")

        # Initialize with default values
        product_data = {"short_caption": "", "long_caption": ""}
        lifestyle_data = {"lifestyle_caption": ""}

        if model_name == "gemini" or model_name == "gemini2":
            model = genai.GenerativeModel(MODELS[model_name].model_id,
                system_instruction= sys_prompt
            
            )
            
            # Configure generation config for JSON output
            generation_config = genai.GenerationConfig(
                temperature=0.1,  # Lower temperature for more deterministic output
            )

            # For Gemini, we need to pass the PIL Image directly
            product_response = model.generate_content(
                [product_prompt, product_image],
                generation_config=generation_config
            )
            if product_response.text:
                try:
                    product_data = json.loads(product_response.text)
                except json.JSONDecodeError:
                    product_data = extract_json_from_text(product_response.text)
            
            if lifestyle_image:
                lifestyle_response = model.generate_content(
                    [lifestyle_prompt, lifestyle_image],
                    generation_config=generation_config
                )
                if lifestyle_response.text:
                    try:
                        lifestyle_data = json.loads(lifestyle_response.text)
                    except json.JSONDecodeError:
                        lifestyle_data = extract_json_from_text(lifestyle_response.text)
                    if not lifestyle_data.get("lifestyle_caption"):
                        lifestyle_data = {"lifestyle_caption": lifestyle_response.text}
        
        elif model_name == "gpt-4o":
            product_response = openai_client.chat.completions.create(
                model=MODELS[model_name].model_id,
                messages=[{
                    "role": "system",
                    "content": sys_prompt
                }, {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": product_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": product_image_url
                            }
                        }
                    ]
                }],
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            content = product_response.choices[0].message.content
            if content:
                product_data = json.loads(content)
            
            if lifestyle_image_url:
                lifestyle_response = openai_client.chat.completions.create(
                    model=MODELS[model_name].model_id,
                    messages=[{
                        "role": "system",
                        "content": sys_prompt
                    }, {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": lifestyle_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": lifestyle_image_url
                                }
                            }
                        ]
                    }],
                    max_tokens=1000,
                    response_format={"type": "json_object"}
                )
                content = lifestyle_response.choices[0].message.content
                if content:
                    lifestyle_data = json.loads(content)
                
        elif model_name == "together":
            product_response = together_client.chat.completions.create(
                model=MODELS[model_name].model_id,
                messages=[{
                    "role": "user",
                    "content": f"{product_prompt}\n\n[Image: {product_image_url}]"
                }]
            )
            content = product_response.choices[0].message.content
            if content:
                product_data = extract_json_from_text(content)
            
            if lifestyle_image_url:
                lifestyle_response = together_client.chat.completions.create(
                    model=MODELS[model_name].model_id,
                    messages=[{
                        "role": "user",
                        "content": f"{lifestyle_prompt}\n\n[Image: {lifestyle_image_url}]"
                    }]
                )
                content = lifestyle_response.choices[0].message.content
                if content:
                    lifestyle_data = extract_json_from_text(content)
                    if not lifestyle_data.get("lifestyle_caption"):
                        lifestyle_data = {"lifestyle_caption": content}
                
        elif model_name == "groq":
            product_response = groq_client.chat.completions.create(
                model=MODELS[model_name].model_id,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": product_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": product_image_url
                            }
                        }
                    ]
                }]
            )
            content = product_response.choices[0].message.content
            if content:
                product_data = extract_json_from_text(content)
            
            if lifestyle_image_url:
                lifestyle_response = groq_client.chat.completions.create(
                    model=MODELS[model_name].model_id,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": lifestyle_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": lifestyle_image_url
                                }
                            }
                        ]
                    }]
                )
                content = lifestyle_response.choices[0].message.content
                if content:
                    lifestyle_data = extract_json_from_text(content)
                    if not lifestyle_data.get("lifestyle_caption"):
                        lifestyle_data = {"lifestyle_caption": content}
                
        elif model_name == "grok" or model_name == "grok2":
            product_response = xai_client.chat.completions.create(
                model=MODELS[model_name].model_id,
                messages=[{
                    "role": "system",
                    "content": sys_prompt
                }, {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": product_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": product_image_url
                            }
                        }
                    ]
                }],
                max_tokens=1000
            )
            content = product_response.choices[0].message.content
            if content:
                product_data = extract_json_from_text(content)
            
            if lifestyle_image_url:
                lifestyle_response = xai_client.chat.completions.create(
                    model=MODELS[model_name].model_id,
                    messages=[{
                        "role": "system",
                        "content": sys_prompt
                    }, {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": lifestyle_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": lifestyle_image_url
                                }
                            }
                        ]
                    }],
                    max_tokens=1000
                )
                content = lifestyle_response.choices[0].message.content
                if content:
                    lifestyle_data = extract_json_from_text(content)
                    if not lifestyle_data.get("lifestyle_caption"):
                        lifestyle_data = {"lifestyle_caption": content}
        
        return {
            f"{model_name}_short_caption": product_data.get("short_caption", ""),
            f"{model_name}_long_caption": product_data.get("long_caption", ""),
            f"{model_name}_lifestyle_caption": lifestyle_data.get("lifestyle_caption", "")
        }
        
    except Exception as e:
        st.error(f"Error with {model_name}: {str(e)}")
        return {
            f"{model_name}_short_caption": "",
            f"{model_name}_long_caption": "",
            f"{model_name}_lifestyle_caption": ""
        }


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page", ["Caption Generator", "Product Scraper"])
    
    if page == "Caption Generator":
        # Your existing caption generator code
        st.title("Product Caption Generator")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx'])
        
        if uploaded_file:
            try:
                # Read Excel file
                df = pd.read_excel(uploaded_file)
                df = df.dropna(how='all')  # Remove completely empty rows
                
                if df.empty:
                    st.error("The uploaded Excel file is empty or contains no valid data.")
                    return
                
                required_columns = ['Image1_Link (Product Image)']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    st.error(f"Missing required columns: {', '.join(missing_columns)}")
                    return
                
                # Process each row
                if st.button("Generate Captions"):
                    with st.spinner("Generating captions..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        total_rows = len(df)
                        
                        # No need to prepare model-specific columns anymore
                        #df['Image1_Type'] = ''
                        df['Image1_Short_Caption'] = ''
                        df['Image1_Long_Caption'] = ''
                        df['Image2_Long_Caption'] = ''
                        
                        for idx, (index, row) in enumerate(df.iterrows()):
                            status_text.write(f"Processing row {idx + 1}/{total_rows}")
                            
                            # Create columns for images
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("Product Image:")
                                product_image = download_image(row['Image1_Link (Product Image)'])
                                if product_image:
                                    st.image(product_image, use_container_width=True)
                            
                            with col2:
                                st.write("Lifestyle Image:")
                                lifestyle_image = download_image(row.get('Image2_link (Lifestyle)', ''))
                                if lifestyle_image:
                                    st.image(lifestyle_image, use_container_width=True)
                            
                            if product_image:
                                success = False
                                error_messages = []
                                
                                # Try each model sequentially until one succeeds
                                for model_name in MODELS:
                                    try:
                                        captions = generate_captions_with_model(
                                            model_name,
                                            row['Image1_Link (Product Image)'],
                                            row.get('Image2_link (Lifestyle)', ''),
                                            row.get('Product Link', row['Image1_Link (Product Image)'])
                                        )
                                        
                                        # Validate that we actually got captions
                                        short_caption = captions.get(f'{model_name}_short_caption', '').strip()
                                        long_caption = captions.get(f'{model_name}_long_caption', '').strip()
                                        lifestyle_caption = captions.get(f'{model_name}_lifestyle_caption', '').strip()
                                        
                                        if not short_caption or not long_caption:
                                            raise Exception(f"No valid captions generated by {model_name}")
                                        
                                        # Update DataFrame with results from the successful model
                                        #df.at[index, 'Image1_Type'] = 'Product'
                                        df.at[index, 'Image1_Short_Caption'] = short_caption
                                        df.at[index, 'Image1_Long_Caption'] = long_caption
                                        if lifestyle_caption:
                                            df.at[index, 'Image2_Long_Caption'] = lifestyle_caption
                                        
                                        success = True
                                        break
                                        
                                    except Exception as e:
                                        error_msg = str(e)
                                        # Check for specific error messages
                                        if any(err in error_msg.lower() for err in [
                                            'resource has been exhausted',
                                            'too many requests',
                                            'response.candidates is empty',
                                            'no valid captions generated'
                                        ]):
                                            error_messages.append(f"{model_name}: {error_msg}")
                                            continue
                                        else:
                                            # For unexpected errors, show them immediately
                                            st.error(f"Unexpected error with {model_name}: {error_msg}")
                                            error_messages.append(f"{model_name}: {error_msg}")
                                            continue
                                
                                if not success:
                                    st.error(f"Error processing row {idx + 1}. All models failed:\n" + "\n".join(error_messages))
                            
                            progress_bar.progress(float(idx + 1) / total_rows)
                        
                        # Display results in data editor
                        st.subheader("Generated Captions")
                        edited_df = st.data_editor(df)
                        
                        # Save to Excel
                        excel_buffer = io.BytesIO()
                        edited_df.to_excel(excel_buffer, index=False)
                        excel_data = excel_buffer.getvalue()
                        
                        # Generate PDF report
                        status_text.write("Generating PDF report...")
                        
                        try:
                            # Convert DataFrame to list of dictionaries for PDF generation
                            captions_data = []
                            for _, row in edited_df.iterrows():
                                captions_data.append({
                                    'product_image_url': row['Image1_Link (Product Image)'],
                                    'lifestyle_image_url': row.get('Image2_link (Lifestyle)', ''),
                                    'short_caption': row['Image1_Short_Caption'],
                                    'long_caption': row['Image1_Long_Caption'],
                                    'lifestyle_caption': row['Image2_Long_Caption']
                                })
                            
                            # Generate and offer PDF download
                            pdf_buffer = io.BytesIO()
                            create_caption_pdf(captions_data, pdf_buffer)
                            pdf_data = pdf_buffer.getvalue()
                            
                            # Create a ZIP file containing both Excel and PDF
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                # Add Excel file
                                zip_file.writestr('captions.xlsx', excel_data)
                                # Add PDF file
                                zip_file.writestr('captions.pdf', pdf_data)
                            
                            # Offer combined download
                            st.download_button(
                                label="Download Reports (Excel + PDF)",
                                data=zip_buffer.getvalue(),
                                file_name="caption_reports.zip",
                                mime="application/zip"
                            )
                            
                            status_text.write("✅ Caption generation complete! Click the button above to download your reports.")
                            
                        except Exception as e:
                            st.error(f"Error generating PDF: {str(e)}")
                        
            except Exception as e:
                st.error(f"Error reading Excel file: {str(e)}")
        pass
    else:
        display_scraper_page()


if __name__ == "__main__":
    main()