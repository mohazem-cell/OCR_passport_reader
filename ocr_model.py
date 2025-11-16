import base64
import requests
import json
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io
from typing import Optional, Dict, Any

class EnhancedOCR:
    def __init__(self, api_url: str, model: str = "unsloth/Qwen2.5-VL-72B-Instruct-bnb-4bit"):
        self.api_url = api_url
        self.model = model
    
    def preprocess_image(self, image_path: str, 
                         resize_factor: float = 2.0,
                         denoise: bool = True,
                         enhance_contrast: bool = True,
                         sharpen: bool = True,
                         binarize: bool = False) -> str:
        """
        Preprocess image to improve OCR quality
        
        Args:
            image_path: Path to input image
            resize_factor: Scale factor (>1 increases resolution)
            denoise: Apply denoising
            enhance_contrast: Enhance image contrast
            sharpen: Apply sharpening
            binarize: Convert to binary (black/white)
        
        Returns:
            Base64 encoded preprocessed image
        """
        # Read image
        img = cv2.imread(image_path)
        
        # 1. Resize for better quality
        if resize_factor != 1.0:
            new_width = int(img.shape[1] * resize_factor)
            new_height = int(img.shape[0] * resize_factor)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # 2. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 3. Denoise
        if denoise:
            gray = cv2.fastNlMeansDenoising(gray, h=10)
        
        # 4. Enhance contrast using CLAHE
        if enhance_contrast:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        
        # 5. Sharpen
        if sharpen:
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            gray = cv2.filter2D(gray, -1, kernel)
        
        # 6. Binarization (optional - good for clean text)
        if binarize:
            gray = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        
        # Convert back to PIL Image for encoding
        pil_img = Image.fromarray(gray)
        
        # Further enhancement with PIL
        if enhance_contrast and not binarize:
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(1.5)
        
        # Encode to base64
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return img_base64
    
    def create_advanced_prompt(self, 
                              ocr_type: str = "general",
                              output_format: str = "markdown") -> str:
        """
        Create optimized prompts for different OCR scenarios
        
        Args:
            ocr_type: Type of OCR task (general, table, form, handwritten, dense)
            output_format: Desired output format (markdown, json, plain)
        
        Returns:
            Optimized prompt string
        """
        base_instructions = """You are an expert OCR system. Your task is to extract ALL text from this image with PERFECT accuracy."""
        
        prompts = {
            "general": f"""{base_instructions}

                    **Instructions:**
                    1. Read the image carefully from top to bottom, right to left
                    2. Extract EVERY piece of text, including:
                    - Headers and titles
                    - Body text and paragraphs
                    - Small print, footnotes, watermarks
                    - Numbers, dates, codes
                    - Any visible labels or captions
                    3. Preserve the original text EXACTLY as written (spelling, punctuation, capitalization)
                    4. Maintain spatial relationships and text grouping
                    5. If text is unclear, provide your best interpretation and mark with [uncertain]

                    **Output format ({output_format}):**
                    - Use clear section separators
                    - Indicate text regions (header, body, footer, etc.)
                    - Preserve line breaks and spacing where meaningful
                    - Format lists, numbered items clearly
                    """,
            
            "table": f"""{base_instructions}

                    **Instructions:**
                    1. Identify all tables in the image
                    2. Extract table structure preserving rows and columns
                    3. Include headers, data cells, and any merged cells
                    4. Capture any table titles or captions
                    5. Note any footnotes or annotations

                    **Output format (Markdown table):**
                    ```
                    | Column 1 | Column 2 | Column 3 |
                    |----------|----------|----------|
                    | Data     | Data     | Data     |
                    ```
                    """,
            
            "form": f"""{base_instructions}

                    **Instructions:**
                    1. Identify all form fields and labels
                    2. Extract field names and their corresponding values
                    3. Capture checkboxes (mark as [X] or [ ])
                    4. Note any instructions or help text
                    5. Preserve the form structure

                    **Output format (JSON):**
                    ```json
                    {{
                    "form_title": "...",
                    "fields": [
                        {{"label": "Name", "value": "..."}},
                        {{"label": "Date", "value": "..."}}
                    ]
                    }}
                    ```
                    """,
            
            "handwritten": f"""{base_instructions}

                    **Special considerations for handwritten text:**
                    1. Text may be cursive or printed
                    2. Take extra care with ambiguous characters (a/o, 1/l, 0/O)
                    3. If a word is illegible, mark as [illegible]
                    4. Provide confidence level: [high], [medium], [low]

                    **Output format:**
                    - Transcribe line by line
                    - Mark uncertain words with [uncertain: possible_word]
                    """,
            
            "dense": f"""{base_instructions}

                    **Instructions for dense/complex documents:**
                    1. Scan systematically: top→bottom, left→right
                    2. Use logical sections (don't merge unrelated text)
                    3. Identify multiple columns and handle separately
                    4. Capture all elements: text, numbers, symbols
                    5. Note document structure (paragraphs, sections, etc.)

                    **Output with clear hierarchy:**
                    - Use headers (##, ###) for sections
                    - Separate columns with clear markers
                    - Maintain reading order
                    """
                            }
        
        return prompts.get(ocr_type, prompts["general"])
    
    def perform_ocr(self, 
                   image_path: str,
                   ocr_type: str = "general",
                   output_format: str = "markdown",
                   preprocess: bool = True,
                   **preprocess_kwargs) -> Dict[str, Any]:
        """
        Perform OCR with preprocessing and advanced prompting
        
        Args:
            image_path: Path to image file
            ocr_type: Type of OCR (general, table, form, handwritten, dense)
            output_format: Output format (markdown, json, plain)
            preprocess: Whether to apply preprocessing
            **preprocess_kwargs: Additional preprocessing parameters
        
        Returns:
            Dictionary with OCR results and metadata
        """
        try:
            # Preprocess image
            if preprocess:
                print("🔄 Preprocessing image...")
                image_base64 = self.preprocess_image(image_path, **preprocess_kwargs)
            else:
                print("📁 Loading image without preprocessing...")
                with open(image_path, "rb") as f:
                    image_base64 = base64.b64encode(f.read()).decode("utf-8")
            
            # Create optimized prompt
            prompt = self.create_advanced_prompt(ocr_type, output_format)
            
            # Prepare payload
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                        ],
                    }
                ],
                "max_tokens": 2000,  # Increased for better coverage
                "temperature": 0.1,   # Low temperature for accuracy
            }
            
            # Send request
            print("🚀 Sending request to model...")
            response = requests.post(
                self.api_url, 
                headers={"Content-Type": "application/json"}, 
                data=json.dumps(payload),
                timeout=180
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract text from response
            extracted_text = result['choices'][0]['message']['content']
            
            return {
                "success": True,
                "text": extracted_text,
                "metadata": {
                    "model": self.model,
                    "ocr_type": ocr_type,
                    "preprocessed": preprocess,
                    "tokens_used": result.get('usage', {})
                },
                "raw_response": result
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"API Error: {str(e)}",
                "text": None
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Processing Error: {str(e)}",
                "text": None
            }


# ============= USAGE EXAMPLES =============

if __name__ == "__main__":
    # Initialize OCR system
    ocr = EnhancedOCR(
        api_url="http://158.176.194.169:4005/v1/chat/completions"
    )
    
    # Example 1: General document with preprocessing
    result = ocr.perform_ocr(
        image_path=r"./OMAR.pdf",
        ocr_type="table",
        output_format="markdown",
        preprocess=True,
        resize_factor=1.5,      # Upscale 2x
        denoise=True,
        enhance_contrast=True,
        sharpen=True,
        binarize=False          # Set True for clean printed text
    )
    
    if result["success"]:
        print("\n" + "="*60)
        print("✅ OCR SUCCESSFUL")
        print("="*60)
        print(result["text"])
        print("\n" + "="*60)
        print(f"📊 Tokens used: {result['metadata']['tokens_used']}")
    else:
        print(f"\n❌ OCR FAILED: {result['error']}")