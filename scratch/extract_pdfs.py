import os
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error reading {pdf_path}: {e}"

def main():
    base_dir = r"e:\crop-disease-outbreak-prediction-system-feature-zip-changes\crop-disease-outbreak-prediction-system-feature-zip-changes\model_train\research_comp\evidence_base\literature\red_rot"
    output_dir = r"e:\crop-disease-outbreak-prediction-system-feature-zip-changes\crop-disease-outbreak-prediction-system-feature-zip-changes\model_train\research_comp\evidence_base\literature\extracted_text"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for filename in os.listdir(base_dir):
        if filename.endswith(".pdf"):
            print(f"Extracting {filename}...")
            pdf_path = os.path.join(base_dir, filename)
            text = extract_text_from_pdf(pdf_path)
            
            txt_filename = filename.replace(".pdf", ".txt")
            with open(os.path.join(output_dir, txt_filename), "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Done: {txt_filename}")

if __name__ == "__main__":
    main()
