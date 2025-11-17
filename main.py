from EnhancedOCR import EnhancedOCR
from dotenv import load_dotenv
import os

if __name__ == "__main__":
    # Initialize OCR system
    # load_dotenv()
    # api_key = os.getenv('API_KEY')
    ocr = EnhancedOCR(
        api_url="http://158.176.194.169:4005/v1/chat/completions"
    )
    
    # Example 1: General document with preprocessing
    result = ocr.perform_ocr(
        image_path=r"./image.jpeg",
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