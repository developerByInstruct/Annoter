# Product Caption Generator

A Streamlit app that generates product and lifestyle captions for images using multiple AI models including OpenAI, Google Gemini, Together, Groq, and Grok.

## Features
- Upload Excel files containing product and lifestyle image URLs
- Automatically download and display images for review
- Generate AI-powered captions using multiple models:
  - OpenAI
  - Google Gemini
  - Together AI
  - Groq
  - Grok
- Generate different types of captions:
  - Short caption for product images (5-15 words)
  - Detailed caption for product images
  - Lifestyle caption for lifestyle images
- Interactive data editor for reviewing and editing captions
- Generate PDF reports with side-by-side comparisons of captions from different models
- Download the updated Excel file with generated captions

## Requirements
- Python 3.7+
- API keys for the AI models you want to use

## Installation
1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your API keys:
```
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here
GROQ_API_KEY=your_groq_key_here
GROK_API_KEY=your_grok_key_here
TOGETHER_API_KEY=your_together_key_here
```

## Usage
1. Prepare your Excel file with the following columns:
   - `Brand_URL`: URL of the brand website
   - `Product_URL`: URL of the product page
   - `Image1_Link (Product Image)`: URL of the product image
   - `Image1_Type`: Type of the product image
   - `Image1_Short_Caption`: Will be filled with generated short caption
   - `Image1_Long_Caption`: Will be filled with generated detailed caption
   - `Image2_link (Lifestyle)`: URL of the lifestyle image
   - `Image2_Long_Caption`: Will be filled with generated lifestyle caption

2. Run the app:
```bash
streamlit run app.py
```

3. Use the app:
   - Upload your Excel file
   - Select which AI models to use for caption generation
   - Click "Generate Captions" to process the images
   - Review the generated captions in the interactive data editor
   - Generate PDF reports to compare captions from different models
   - Download the updated Excel file with all generated captions

## PDF Reports
The app generates detailed PDF reports that include:
- Product images with captions from each selected AI model
- Lifestyle images with their respective captions
- Side-by-side comparison of captions for easy evaluation
- Organized sections for short captions, long captions, and lifestyle captions

## Contributing
Feel free to open issues or submit pull requests for any improvements.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
