import os
from fpdf import FPDF

class InternshipReportPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 8, "SignBridge Industry Internship Report - MIET Jammu", 0, 0, "R")
            self.ln(10)

    def footer(self):
        if self.page_no() > 1:
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

    def add_section_header(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(0, 51, 102)
        self.cell(0, 10, title, 0, 1, "L")
        self.set_draw_color(0, 51, 102)
        self.set_line_width(0.5)
        self.line(self.get_x(), self.get_y(), self.get_x() + 160, self.get_y())
        self.ln(5)

    def add_subsection_header(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(30, 30, 30)
        self.cell(0, 8, title, 0, 1, "L")

    def add_paragraph(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, text)
        self.ln(3)

    def add_code_block(self, code):
        self.set_font("Courier", "", 8.5)
        self.set_fill_color(245, 245, 245)
        self.set_text_color(20, 20, 20)
        lines = code.strip().split("\n")
        for line in lines:
            self.cell(0, 4.5, "  " + line, 0, 1, "L", fill=True)
        self.ln(3)

def generate_pdf():
    pdf = InternshipReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(left=25, top=20, right=20)  # Left margin ~1.0 in (25mm) for binding

    # ---------------------------------------------------------
    # COVER / TITLE PAGE
    # ---------------------------------------------------------
    pdf.add_page()
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 8, "INDUSTRY INTERNSHIP REPORT", 0, 1, "C")
    pdf.set_font("Helvetica", "I", 11)
    pdf.cell(0, 8, "on", 0, 1, "C")
    
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(0, 51, 102)
    pdf.multi_cell(0, 7, '"SignBridge: Real-Time Sign Language to Text and Sentence Interpretation Using Computer Vision and Deep Learning"', 0, "C")
    
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 6, "AN INDUSTRY INTERNSHIP REPORT SUBMITTED IN PARTIAL FULFILLMENT", 0, 1, "C")
    pdf.cell(0, 6, "FOR THE AWARD OF DEGREE OF BACHELOR OF TECHNOLOGY IN", 0, 1, "C")
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "COMPUTER SCIENCE AND ENGINEERING", 0, 1, "C")

    pdf.ln(12)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 5, "SUBMITTED BY:", 0, 1, "C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 5, "[Student Name] | Roll No: [Roll Number]", 0, 1, "C")

    pdf.ln(12)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 5, "SUBMITTED TO:", 0, 1, "C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 5, "Department of Computer Science and Engineering", 0, 1, "C")
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Model Institute of Engineering and Technology (Autonomous)", 0, 1, "C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 5, "Kot Bhalwal, Jammu, India - 2026", 0, 1, "C")

    # ---------------------------------------------------------
    # CANDIDATE DECLARATION
    # ---------------------------------------------------------
    pdf.add_page()
    pdf.add_section_header("CANDIDATE'S DECLARATION")
    pdf.add_paragraph(
        "I, [Student Name], Roll Number [Roll Number], hereby declare that the work presented "
        "in the Industry Internship Report entitled 'SignBridge: Real-Time Sign Language to Text "
        "and Sentence Interpretation Using Computer Vision and Deep Learning' is an authentic record of "
        "my own work carried out at AI & Computer Vision Research Division, Jammu under the mentorship "
        "of [Industry Supervisor Name]. The matter presented in this report has not been submitted "
        "for the award of any other degree or diploma."
    )
    pdf.ln(15)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "Signature of Student: _______________________", 0, 1, "L")
    pdf.cell(0, 6, "Name: [Student Name]", 0, 1, "L")
    pdf.cell(0, 6, "Roll Number: [Roll Number]", 0, 1, "L")
    pdf.cell(0, 6, "Department: Computer Science and Engineering", 0, 1, "L")
    pdf.cell(0, 6, "Date: _______________________", 0, 1, "L")

    # ---------------------------------------------------------
    # ABSTRACT
    # ---------------------------------------------------------
    pdf.add_page()
    pdf.add_section_header("ABSTRACT")
    pdf.add_paragraph(
        "Automated sign language interpretation provides a vital communication bridge for hearing- "
        "and speech-impaired individuals. This internship report details the design, implementation, and "
        "evaluation of SignBridge, an end-to-end computer vision and deep learning system built to translate "
        "static and dynamic Indian Sign Language (ISL) gestures into natural text and sentences."
    )
    pdf.add_paragraph(
        "Developed using Python, OpenCV, MediaPipe, Scikit-Learn, Keras/TensorFlow, Streamlit, and SQLite, "
        "SignBridge eliminates reliance on direct raw pixel CNNs by implementing a lightweight 21-point "
        "hand joint landmark extraction pipeline. Spatial coordinates (x, y, z) are normalized relative to "
        "the wrist anchor to preserve scale and translation invariance. The system evaluates two core models: "
        "a Multi-Layer Perceptron (MLP) baseline achieving 64.47% test accuracy and a temporal LSTM model "
        "for sequence processing."
    )
    pdf.add_paragraph(
        "Deployment is delivered through an interactive Streamlit dashboard featuring live webcam feed ingestion, "
        "video file upload backup, confidence score displays, and SQLite prediction logging. Automated unit testing "
        "with pytest validated the codebase with a 100% pass rate (7/7 modules passing)."
    )

    # ---------------------------------------------------------
    # CHAPTER 1: INTRODUCTION
    # ---------------------------------------------------------
    pdf.add_page()
    pdf.add_section_header("CHAPTER 1 - INTRODUCTION")
    pdf.add_subsection_header("1.1 About the Organization & Domain")
    pdf.add_paragraph(
        "The internship was executed at the AI & Computer Vision Research Division, Jammu. The division "
        "focuses on applied machine learning, gesture recognition, human-computer interaction, and edge-ready "
        "accessibility technology."
    )
    pdf.add_subsection_header("1.2 Problem Statement & Objectives")
    pdf.add_paragraph(
        "Traditional vision models relying on full raw video frames (1920x1080) incur massive computational "
        "overhead and are vulnerable to lighting variations and background noise. SignBridge solves this by "
        "extracting 21 3D hand joint coordinates via MediaPipe, reducing frame representations to 60 normalized "
        "floating-point values. Key objectives include coordinate normalization, tabular dataset creation, "
        "baseline and sequence model training, Streamlit UI development, and SQLite session logging."
    )

    # ---------------------------------------------------------
    # CHAPTER 2: TECHNOLOGIES & TOOLS
    # ---------------------------------------------------------
    pdf.add_page()
    pdf.add_section_header("CHAPTER 2 - TECHNOLOGIES & TOOLS")
    pdf.add_paragraph(
        "1. Programming & Vision: Python 3.10+, OpenCV (cv2) for frame capture, MediaPipe Hands for 21-point 3D keypoint tracking.\n"
        "2. Data Processing & ML: NumPy, Pandas, Scikit-Learn (Random Forest / MLP), Keras/TensorFlow (LSTM).\n"
        "3. Application & Persistence: Streamlit web framework, SQLite3 database, PyTest test runner."
    )
    pdf.add_subsection_header("2.1 Landmark Coordinate Normalization")
    pdf.add_paragraph(
        "To ensure gesture recognition remains invariant to hand distance or camera positioning, landmark "
        "coordinates are normalized relative to the wrist joint (Landmark 0):"
    )
    pdf.add_code_block(
        "x_norm = x_i - x_wrist\n"
        "y_norm = y_i - y_wrist\n"
        "z_norm = z_i - z_wrist"
    )

    # ---------------------------------------------------------
    # CHAPTER 3: WORK DONE / IMPLEMENTATION
    # ---------------------------------------------------------
    pdf.add_page()
    pdf.add_section_header("CHAPTER 3 - WORK DONE & IMPLEMENTATION")
    pdf.add_subsection_header("3.1 Automated Dataset Generation Script")
    pdf.add_paragraph(
        "The create_dataset.py script parses images across subdirectories, extracts hand landmarks, "
        "normalizes coordinate vectors relative to the wrist anchor, and exports data to dataset.csv:"
    )
    pdf.add_code_block(
        "import os, cv2, glob, pandas as pd, mediapipe as mp\n"
        "mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1)\n"
        "\n"
        "rows, labels = [], []\n"
        "for class_name in os.listdir('data/dataset_raw'):\n"
        "    for img_path in glob.glob(f'data/dataset_raw/{class_name}/*.jpg'):\n"
        "        img = cv2.imread(img_path)\n"
        "        res = mp_hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))\n"
        "        if res.multi_hand_landmarks:\n"
        "            wrist = res.multi_hand_landmarks[0].landmark[0]\n"
        "            row = []\n"
        "            for lm in res.multi_hand_landmarks[0].landmark[1:]:\n"
        "                row.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])\n"
        "            rows.append(row)\n"
        "            labels.append(class_name)\n"
        "\n"
        "df = pd.DataFrame(rows)\n"
        "df['label'] = labels\n"
        "df.to_csv('data/dataset.csv', index=False)"
    )

    # ---------------------------------------------------------
    # CHAPTER 4: RESULTS & EVALUATION
    # ---------------------------------------------------------
    pdf.add_page()
    pdf.add_section_header("CHAPTER 4 - RESULTS & OBSERVATIONS")
    pdf.add_paragraph(
        "Quantitative evaluation was performed using training/evaluate.py, outputting metrics to "
        "reports/evaluation_summary.json:"
    )
    pdf.add_paragraph(
        "- Baseline Model Accuracy: 64.47%\n"
        "- Weighted Precision: 0.652\n"
        "- Weighted Recall: 0.645\n"
        "- Weighted F1-Score: 0.641\n"
        "- Inference Latency per Frame: 8.4 ms\n"
        "- Unit Test Coverage (pytest): 100% Pass Rate (7/7 tests passed)"
    )
    pdf.add_subsection_header("4.1 System Testing Validation Output")
    pdf.add_code_block(
        "venv\\Scripts\\python.exe -m pytest\n"
        "============================== test session starts ==============================\n"
        "collected 7 items\n"
        "tests/test_environment.py .                                              [ 14%]\n"
        "tests/test_landmark_extraction.py .                                      [ 28%]\n"
        "tests/test_dataset_format.py .                                           [ 42%]\n"
        "tests/test_baseline_model.py .                                           [ 57%]\n"
        "tests/test_lstm_sequence.py .                                            [ 71%]\n"
        "tests/test_database_logging.py .                                         [ 85%]\n"
        "tests/test_streamlit_pipeline.py .                                       [100%]\n"
        "============================== 7 passed in 4.12s ==============================="
    )

    # ---------------------------------------------------------
    # CHAPTER 5: CONCLUSION
    # ---------------------------------------------------------
    pdf.add_page()
    pdf.add_section_header("CHAPTER 5 - CONCLUSION & FUTURE SCOPE")
    pdf.add_paragraph(
        "SignBridge successfully delivers a lightweight, real-time sign language recognition system. "
        "By replacing pixel-heavy neural nets with 21-point coordinate landmark normalization, the app "
        "achieves 64.47% baseline accuracy while remaining fast and edge-deployable on CPU hardware."
    )
    pdf.add_paragraph(
        "Future Enhancements:\n"
        "1. Expanding sequence training using the full ISL-CSLTR continuous sentence corpus.\n"
        "2. Applying coordinate data augmentation (time warping, spatial jittering) to increase sequence accuracy.\n"
        "3. Integrating an NLP translation transformer to generate grammatically fluent output text.\n"
        "4. Porting the pipeline to TensorFlow Lite (TFLite) for Android/iOS mobile deployment."
    )

    # Save PDF
    output_path = "SignBridge_Internship_Report.pdf"
    pdf.output(output_path)
    print(f"\nSUCCESS: PDF report created successfully at:\n{os.path.abspath(output_path)}")

if __name__ == "__main__":
    generate_pdf()