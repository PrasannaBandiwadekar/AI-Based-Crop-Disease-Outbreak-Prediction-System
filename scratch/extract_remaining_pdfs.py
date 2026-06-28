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
    folders = [
        r"e:\crop-disease-outbreak-prediction-system-feature-zip-changes\crop-disease-outbreak-prediction-system-feature-zip-changes\model_train\research_comp\evidence_base\literature\climate_disease_relationships",
        r"e:\crop-disease-outbreak-prediction-system-feature-zip-changes\crop-disease-outbreak-prediction-system-feature-zip-changes\model_train\research_comp\evidence_base\literature\sugarcane_disease"
    ]
    output_dir = r"e:\crop-disease-outbreak-prediction-system-feature-zip-changes\crop-disease-outbreak-prediction-system-feature-zip-changes\model_train\research_comp\evidence_base\literature\extracted_text"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for base_dir in folders:
        if not os.path.exists(base_dir): continue
        for root, dirs, files in os.walk(base_dir):
            for filename in files:
                if filename.endswith(".pdf"):
                    print(f"Extracting {filename}...")
                    pdf_path = os.path.join(root, filename)
                    text = extract_text_from_pdf(pdf_path)
                    
                    safe_filename = "".join([c if c.isalnum() or c in "._-" else "_" for c in filename])
                    txt_filename = safe_filename.replace(".pdf", ".txt")
                    if len(txt_filename) > 50:
                        txt_filename = txt_filename[:46] + ".txt"
                    with open(os.path.join(output_dir, txt_filename), "w", encoding="utf-8") as f:
                        f.write(text)
                    print(f"Done: {txt_filename}")

if __name__ == "__main__":
    main()
