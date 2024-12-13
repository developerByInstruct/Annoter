from pdf_generator import create_caption_pdf

# Sample data structure with distinct captions for each model
test_data = [{
    'product_image_url': 'https://www.abbyson.com/cdn/shop/files/2_001ddded-5430-408a-a21b-7b632cdca8f7.png?v=1715017063&width=823',
    'lifestyle_image_url': 'https://www.abbyson.com/cdn/shop/files/Luxe-GRY-2pc-Lifestyle1.png?v=1715017063&width=823',
    
    # Gemini captions
    'gemini_short_caption': 'Grey two-seater sofa with black wooden legs',
    'gemini_long_caption': 'A grey fabric loveseat with modular design, featuring connected sections and plush cushions',
    'gemini_lifestyle_caption': 'Grey sofa in bright living room with large windows',
    
    # OpenAI captions
    'openai_short_caption': 'Modern grey loveseat with plush cushions',
    'openai_long_caption': 'Contemporary two-seater with grey upholstery and dark wooden support',
    'openai_lifestyle_caption': 'Sofa positioned near window in modern living space',
    
    # Together captions
    'together_short_caption': 'Grey fabric sofa with wooden base',
    'together_long_caption': 'Two-section grey sofa with textured upholstery and matching pillows',
    'together_lifestyle_caption': 'Sofa in minimalist room with natural lighting',
    
    # Groq captions
    'groq_short_caption': 'Elegant grey loveseat with modern design',
    'groq_long_caption': 'Contemporary loveseat with grey fabric and black cylindrical legs',
    'groq_lifestyle_caption': 'Sofa showcased in bright living area with windows',
    
    # Grok captions
    'grok_short_caption': 'Modern grey sofa with plush seating',
    'grok_long_caption': 'Contemporary loveseat with grey fabric and wooden support legs',
    'grok_lifestyle_caption': 'Sofa displayed in sunlit room with wooden floors'
}]

# Create PDF
output_path = 'test_output.pdf'
create_caption_pdf(test_data, output_path)
print(f"PDF generated at: {output_path}")
