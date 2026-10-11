import os
import gradio as gr
from logic import detect_violation

TITLE = "🪖 Smart Helmet Violation & Challan Detection System"
SUBTITLE = "Deep Learning Project | Automated Traffic Compliance with YOLOv8 & EasyOCR"
DESCRIPTION = """
This Deep Learning application automatically inspects motorcycle and scooter traffic images:
- **Detects Helmet Compliance:** Identifies whether the rider is wearing a helmet or riding without a helmet.
- **Computes Penalty:** If helmet is detected: **No Fine (₹0)**. If no helmet is detected: **Fine: ₹1000**.
- **Recognizes Number Plate:** Detects the vehicle license plate region and extracts the plate number via OCR.
"""

def create_app():
    # Discover available sample examples
    examples = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(base_dir, "samples")
    
    ex_helmet = os.path.join(sample_dir, "sample_helmet.jpg")
    ex_no_helmet = os.path.join(sample_dir, "sample_no_helmet.jpg")
    
    if os.path.exists(ex_helmet):
        examples.append([ex_helmet, 0.25])
    if os.path.exists(ex_no_helmet):
        examples.append([ex_no_helmet, 0.25])

    with gr.Blocks(title="Helmet Violation Detection System") as demo:
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(f"*{SUBTITLE}*")
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            # Left Column: Inputs & Controls
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="📤 Upload Rider / Traffic Photo",
                    type="numpy",
                    sources=["upload", "clipboard", "webcam"],
                    elem_id="input_image"
                )
                
                conf_slider = gr.Slider(
                    minimum=0.10,
                    maximum=0.90,
                    value=0.25,
                    step=0.05,
                    label="🎯 Model Confidence Threshold",
                    info="Lower values detect smaller/distant objects; higher values ensure high certainty."
                )

                with gr.Row():
                    submit_btn = gr.Button("🚨 Analyze Violation", variant="primary", size="lg")
                    clear_btn = gr.ClearButton(components=[image_input], value="🔄 Clear", size="lg")

                # Built-in 1-Click Examples
                if examples:
                    gr.Markdown("### 💡 Click an Example to Test Instantly:")
                    gr.Examples(
                        examples=examples,
                        inputs=[image_input, conf_slider],
                        label="Sample Test Cases",
                        cache_examples=False
                    )

            # Right Column: Detection Results & Fine Details
            with gr.Column(scale=1):
                annotated_output = gr.Image(
                    label="🖼️ Annotated Deep Learning Output",
                    elem_id="output_image"
                )

                with gr.Row():
                    status_output = gr.Textbox(
                        label="🪖 Helmet Status",
                        interactive=False
                    )
                    fine_output = gr.Textbox(
                        label="💰 Fine Amount",
                        interactive=False
                    )
                    plate_output = gr.Textbox(
                        label="🚗 Vehicle Number Plate",
                        interactive=False
                    )

                report_output = gr.Markdown(
                    value="*Upload an image or select an example on the left to view the violation challan report.*"
                )

        # Technical Project Details Accordion (for Viva / Project grading)
        with gr.Accordion("🧠 Deep Learning Architecture & Project Details", open=False):
            gr.Markdown("""
### Technical Pipeline:
1. **Object Detection (Backbone):**
   - **YOLOv8 Nano (CNN)** trained for headgear recognition:
     - Class 0: `With Helmet`
     - Class 1: `Without Helmet`
   - Real-time bounding box localization using anchor-free CSPDarknet53 feature extraction.
2. **License Plate Detection:**
   - Fine-tuned YOLOv8 detector for segmenting vehicle registration number plates under varying angles and illumination.
3. **Optical Character Recognition (OCR):**
   - **EasyOCR Pipeline:** CRAFT (Character Region Awareness for Text Detection) combined with ResNet-BiLSTM-CTC recognition model.
4. **Traffic Law Enforcement Rule Engine:**
   - Under **Section 129 of the Motor Vehicles Act**, driving a two-wheeler without protective headgear is an offense penalizable with a **₹1,000 fine** and automated challan generation.
            """)

        # Event binding
        submit_btn.click(
            fn=detect_violation,
            inputs=[image_input, conf_slider],
            outputs=[annotated_output, status_output, fine_output, plate_output, report_output]
        )

        clear_btn.add([annotated_output, status_output, fine_output, plate_output, report_output])

    return demo

demo = create_app()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)