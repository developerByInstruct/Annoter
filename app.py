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

# Initialize API clients
genai.configure(api_key=GOOGLE_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
together_client = Together(api_key=TOGETHER_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)
xai_client = OpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1")

sys_prompt = """You are a professional objective product copywriter. You are not subjective, nor speak in a marketing tone. Your task is to:
Provide the short caption and long caption for product image
Provide only a long caption for the lifestyle image.

What is the product image?
The product image should be isolated
Color
Featuring either a transparent or solid color background
A neutral or plain background

Objects in the image
The image never includes humans. Objects in an image could come in pairs/triplets/etc.
Object could be located not in focal point of an image
There should be no text or graphics on the image (Text or graphics printed on the product is OK)
No other artefacts present on image, such as stripes, compression artefacts, marks, or highlights
The image should not have an extreme aspect ratio (>4)
It should not depict the product in a real-life environment or setting. It is merely an image designed to show the user a given product in a high-level of detail from a given angle, or series of angles through multiple product images
The whole product should be shown

Packaging
If the product has packaging, then show the product, not the packaging

Lighting and Colors Requirements
The lighting of an image shouldn’t be too dark or bright that ends up distorting or hiding the original product’s details
Ensure the color of the object is accurate
Example:

To add caption:
For the Short Description
Emphasize simplicity.
Focus only on the key elements. What this object is, if something that comes in different shapes (i.e. perfume bottle), function, material, color
20-25 words maximum, but better to keep it 5-15 words
Avoid using too many details for the short description and focus only on the key attributes.
Example: A cordless vacuum with a light blue base and silver and gold handle

For the Long Description
Include all the specific details: the color of the fan, the number of blades, the material of the blades, brand/manufacturer and any additional features such as the industrial cage around the lights.
Do not describe it in a way to sell it, describe it in a way to help a machine learning model to understand what it looks like. Only include details that can be gathered from looking at the image, do not include details that would only be learned from the product description.
Any positioning should be based on your perspective looking at the image e.g. left, right

Examples
Short caption: A small white jar with blue cover.
Long caption: A white round jar with a blue cover. The jar has a blue lotus flower design with a vertical blue line that runs down the packaging. The text "DERMA - E" is written below in dark color. Below that is "Eczema Relief Cream" in black font color. The size is written as 4 OZ/ 113g at the lower right corner of the jar

What is a lifestyle image?
Setting 
An image that depicts the product in use within a real-life setting, showcasing its practical application and context. 
This type of image helps to convey how the product fits into the daily life of the consumer, often featuring human interaction or a relatable environment. 

Objects in the image 
There should be no text or graphics on the image (Text or graphics printed on the product is OK) 
Object could be located not in focal point of an image 
Photo can feature multiple instances of a product 
No other artefacts present on image such as stripes, compression artefacts, marks, or highlights 
The image should not have an extreme aspect ratio (>4) 
The whole product should be shown - Extra additions to the image that wouldn’t be natural
 
Lighting and Color Requirements 
The lighting of an image shouldn’t be too dark or bright that ends up distorting or hiding the original product’s details 
Ensure the color of the object is accurate

Example:


To add caption:
Identify and describe the main objects and their positions within the image (e.g. centered, top right corner, slightly off center to the left) 
Positions i.e. “left” and “right” are based on your perspective looking at the image
Describe any prominent shapes and colors of these objects. Describe any textures or patterns.
Details of how this product exists and interacts with the rest of the objects in this image
Describe the background in the image. 
Avoid any description speaking to the feeling of the image (e.g. "adding a touch of life to the composition.") or any other commentary that doesn’t help re create the image. (You aren’t trying to sell the product, you’re trying to describe how it looks for a machine learning model)
Pay special attention to ensuring the product image that's now in the lifestyle image is described in a way it can be reproduced with sufficient detail.

Examples
Long Caption: The image shows a woman with shoulder-length blonde hair  holding a jar of skin product. The woman is light-skinned and is smiling broadly, showcasing her teeth. She is wearing a purple t-shirt. The jar she holds is white with teal-colored accents and text. The text on the jar appears to be the brand name "DERMA E" in a sans-serif, capital font. The jar is centered in her palm, and her hand is holding it. The background is a plain, off-white wall.

Remember:
1. Always be objective, and do not use subjective statements like: possibly, maybe, likely, appears to, 
2. Only caption what you see, and do not make inferences from the objective facts
3. Always begin with 'The image...' """

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

def generate_prompt(image_type="product"):
    base_prompt = """

Analyze this {type} image and provide captions in the following JSON format:
{format}"""

    if image_type == "product":
        criteria = """- Focus on physical attributes: dimensions, materials, colors, textures
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

    return base_prompt.format(type=image_type, criteria=criteria, format=json_format)

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

def generate_captions_with_model(model_name, product_image_url, lifestyle_image_url=None):
    product_prompt = generate_prompt("product")
    lifestyle_prompt = generate_prompt("lifestyle")
    
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

        if model_name == "gemini":
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
                
        elif model_name == "grok":
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
                    
                    # Prepare DataFrame for results
                    for model_name in MODELS:
                        df[f'{model_name}_short_caption'] = ''
                        df[f'{model_name}_long_caption'] = ''
                        df[f'{model_name}_lifestyle_caption'] = ''
                    
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
                            try:
                                # Generate captions for each model
                                for model_name in MODELS:
                                    captions = generate_captions_with_model(
                                        model_name,
                                        row['Image1_Link (Product Image)'],
                                        row.get('Image2_link (Lifestyle)', '')
                                    )
                                    
                                    # Update DataFrame with results
                                    for key, value in captions.items():
                                        df.at[index, key] = value
                                
                            except Exception as e:
                                st.error(f"Error processing row {idx + 1}: {str(e)}")
                        
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
                            row_data = {
                                'product_image_url': row['Image1_Link (Product Image)'],
                                'lifestyle_image_url': row['Image2_link (Lifestyle)'],
                            }
                            
                            for model_name in MODELS:
                                row_data.update({
                                    f"{model_name}_short_caption": row.get(f"{model_name}_short_caption", ""),
                                    f"{model_name}_long_caption": row.get(f"{model_name}_long_caption", ""),
                                    f"{model_name}_lifestyle_caption": row.get(f"{model_name}_lifestyle_caption", "")
                                })
                            
                            captions_data.append(row_data)
                        
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

if __name__ == "__main__":
    main()
