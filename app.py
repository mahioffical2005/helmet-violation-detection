import gradio as gr
from logic import detect_violation

TITLE = "🪖 Smart Helmet Violation & Challan Detection System"
SUBTITLE = "Deep Learning Project | Automated Traffic Compliance with YOLOv8 & EasyOCR"
DESCRIPTION = """
Upload a motorcycle/scooter rider photo to run automatic Deep Learning inspection:
- **Helmet Compliance:** Detects whether the rider is wearing a helmet or riding without a helmet.
- **Traffic Penalty:** If helmet is detected: **No Fine (₹0)**. If no helmet is detected: **Fine: ₹1000**.
- **Number Plate:** Automatically extracts vehicle license plate text using OCR.
"""

def create_app():
    with gr.Blocks(title="Helmet Violation Detection System") as demo:
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(f"*{SUBTITLE}*")
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            # Left Column: Upload & Controls (NO EXAMPLES)
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="📤 Upload Rider / Traffic Photo",
                    type="pil",
                    sources=["upload", "clipboard", "webcam"],
                    elem_id="input_image"
                )
                
                conf_slider = gr.Slider(
                    minimum=0.10,
                    maximum=0.90,
                    value=0.20,
                    step=0.05,
                    label="🎯 Model Confidence Threshold",
                    info="Sensitivity for detecting helmet and vehicle."
                )

                with gr.Row():
                    submit_btn = gr.Button("🚨 Analyze Violation", variant="primary", size="lg")
                    clear_btn = gr.ClearButton(components=[image_input], value="🔄 Clear", size="lg")

            # Right Column: Output & Details
            with gr.Column(scale=1):
                annotated_output = gr.HTML(
                    label="🖼️ Annotated Deep Learning Output",
                    value="<div style='display:flex;justify-content:center;align-items:center;height:300px;background:#F9FAFB;border-radius:8px;border:1px dashed #D1D5DB;color:#6B7280;'>Upload image and click Analyze Violation to view detection</div>",
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
                    value="*Upload an image on the left and click **Analyze Violation** to inspect.*"
                )

        # Technical Details Accordion
        with gr.Accordion("🧠 Deep Learning Architecture & Project Details", open=False):
            gr.Markdown("""
### Technical Pipeline:
1. **Object Detection (Backbone):**
   - **YOLOv8 Nano (CNN)** trained for headgear recognition:
     - Class 0: `With Helmet`
     - Class 1: `Without Helmet`
2. **License Plate Detection:**
   - YOLOv8 detector for segmenting vehicle registration number plates under varying angles and illumination.
3. **Optical Character Recognition (OCR):**
   - **EasyOCR Pipeline:** CRAFT (Character Region Awareness for Text Detection) combined with ResNet-BiLSTM-CTC recognition model.
4. **Traffic Law Enforcement Rule Engine:**
   - Under **Section 129 of the Motor Vehicles Act**, riding without a helmet incurs an automated penalty of **₹1,000 fine** and e-challan generation.
            """)

        # Event binding with show_api=False to avoid schema serialization bug
        submit_btn.click(
            fn=detect_violation,
            inputs=[image_input, conf_slider],
            outputs=[annotated_output, status_output, fine_output, plate_output, report_output],
            api_name=False,
            show_api=False
        )

        clear_btn.add([annotated_output, status_output, fine_output, plate_output, report_output])

    return demo

demo = create_app()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)