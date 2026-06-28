import os

def rename_red_rot():
    pdf_dir = r"e:\crop-disease-outbreak-prediction-system-feature-zip-changes\crop-disease-outbreak-prediction-system-feature-zip-changes\model_train\research_comp\evidence_base\literature\applied_management"
    long_pdf_dir = "\\\\?\\" + os.path.abspath(pdf_dir)
    try:
        files = os.listdir(long_pdf_dir)
        print(f"Files in directory: {files}")
        for f in files:
            if "Red Rot" in f and f.endswith(".pdf"):
                old_path = os.path.join(long_pdf_dir, f)
                new_path = os.path.join(long_pdf_dir, "red_rot_mgmt_2022.pdf")
                os.rename(old_path, new_path)
                print(f"Renamed '{f}' to 'red_rot_mgmt_2022.pdf'")
                return True
        print("No Red Rot file found.")
    except Exception as e:
        print(f"Error: {e}")
    return False

if __name__ == "__main__":
    rename_red_rot()
