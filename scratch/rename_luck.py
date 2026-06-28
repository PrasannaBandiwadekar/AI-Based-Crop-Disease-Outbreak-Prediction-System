import os

def rename_luck():
    pdf_dir = r"e:\crop-disease-outbreak-prediction-system-feature-zip-changes\crop-disease-outbreak-prediction-system-feature-zip-changes\model_train\research_comp\evidence_base\literature\climate_disease_relationships"
    long_pdf_dir = "\\\\?\\" + os.path.abspath(pdf_dir)
    try:
        files = os.listdir(long_pdf_dir)
        print(f"Files in directory: {files}")
        for f in files:
            if "Luck" in f and f.endswith(".pdf"):
                old_path = os.path.join(long_pdf_dir, f)
                new_path = os.path.join(long_pdf_dir, "luck_2011.pdf")
                os.rename(old_path, new_path)
                print(f"Renamed '{f}' to 'luck_2011.pdf'")
                return True
        print("No Luck file found.")
    except Exception as e:
        print(f"Error: {e}")
    return False

if __name__ == "__main__":
    rename_luck()
