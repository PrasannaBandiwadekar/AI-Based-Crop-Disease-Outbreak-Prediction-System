import os
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path, output_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Successfully extracted to {output_path}")
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")

def main():
    pdf_dir = r"e:\crop-disease-outbreak-prediction-system-feature-zip-changes\crop-disease-outbreak-prediction-system-feature-zip-changes\model_train\research_comp\evidence_base\literature\applied_management"
    pdf_name = "red_rot_mgmt_2022.pdf"
    pdf_path = os.path.join(pdf_dir, pdf_name)
    
    output_dir = r"e:\crop-disease-outbreak-prediction-system-feature-zip-changes\crop-disease-outbreak-prediction-system-feature-zip-changes\model_train\research_comp\evidence_base\literature\extracted_text"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_name = "red_rot_mgmt_2022_extracted.txt"
    output_path = os.path.join(output_dir, output_name)
    
    extract_text_from_pdf(pdf_path, output_path)

if __name__ == "__main__":
    main()
