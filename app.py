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

# Initialize API clients
genai.configure(api_key=GOOGLE_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
together_client = Together(api_key=TOGETHER_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)
xai_client = OpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1")

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
    base_prompt = """You are a professional objective product copywriter. You are not subjective, nor speak in a marketing tone. Your task is to analyze {type} images and provide appropriate captions based on the following examples:

Example Product Image:
Brand: Abbyson
Product: Luxe Stain Resistant Fabric 2pc Sofa
Short Caption: A grey two-seater sofa with black wooden legs and cushioned seating.
Long Caption: A grey fabric loveseat with a modular design. The loveseat consists of two connected sections, each with a seat cushion, creating a slight dip in the middle. The upholstery has a textured appearance and is made of woven fabric. The backrest cushions are large and loose and the colour matches the seat cushion's fabric. Two additional square throw pillows rest against each arm, mirroring the design of the back cushions. The loveseat sits on four short, cylindrical black legs, with small space between the base and the floor.

Example Lifestyle Image:
Brand: Philosophy
Product: Ultimate Miracle Worker Body Serum
Lifestyle Caption: A bottle of "ultimate miracle worker" by Philosophy is resting on the inner part of a person's elbow, which is curved to form a V shape, with the bottle nestled in the crook. The person's knees are also visible, folded upwards as if they are seated, with only the kneecaps showing. The white bottle features a black pump dispenser and a label with multiple lines of text. The front of the label prominently displays the product name, "ultimate miracle worker," along with the brand name, "philosophy."

What to look for in a {type} image:
{criteria}

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
    "short_caption": "5-15 words focusing on key elements (object type, function, material, color)",
    "long_caption": "Detailed description of all visual elements, focusing on physical attributes visible in the image. Include materials, colors, textures, and design features."
}"""
    else:  # lifestyle
        criteria = """- Describe the product's placement and context
- Note any human interaction or positioning
- Detail the environmental setting
- Describe lighting and atmosphere
- Include relevant background elements
- Focus on how the product is being used or displayed"""
        
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
            model = genai.GenerativeModel(MODELS[model_name].model_id)
            
            # For Gemini, we need to pass the PIL Image directly
            product_response = model.generate_content([product_prompt, product_image])
            if product_response.text:
                product_data = extract_json_from_text(product_response.text)
            
            if lifestyle_image:
                lifestyle_response = model.generate_content([lifestyle_prompt, lifestyle_image])
                if lifestyle_response.text:
                    lifestyle_data = extract_json_from_text(lifestyle_response.text)
                    if not lifestyle_data.get("lifestyle_caption"):
                        lifestyle_data = {"lifestyle_caption": lifestyle_response.text}
        
        elif model_name == "openai":
            product_response = openai_client.chat.completions.create(
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
                }],
                max_tokens=1000
            )
            content = product_response.choices[0].message.content
            if content:
                product_data = extract_json_from_text(content)
            
            if lifestyle_image_url:
                lifestyle_response = openai_client.chat.completions.create(
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
                    }],
                    max_tokens=1000
                )
                content = lifestyle_response.choices[0].message.content
                if content:
                    lifestyle_data = extract_json_from_text(content)
                    if not lifestyle_data.get("lifestyle_caption"):
                        lifestyle_data = {"lifestyle_caption": content}
                
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
                    
                    st.download_button(
                        label="Download Updated Excel",
                        data=excel_data,
                        file_name="updated_captions.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
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
                        
                        st.download_button(
                            label="Download Captions PDF",
                            data=pdf_data,
                            file_name="captions_comparison.pdf",
                            mime="application/pdf"
                        )
                        
                        status_text.write("✅ Caption generation complete! Click the buttons above to download your reports.")
                        
                    except Exception as e:
                        st.error(f"Error generating PDF: {str(e)}")
                    
        except Exception as e:
            st.error(f"Error reading Excel file: {str(e)}")

if __name__ == "__main__":
    main()
