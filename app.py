"""
Resume AI — Gradio web app

Features:
  1. Scrape a job posting URL → extract structured job details
  2. Paste raw job description text → extract structured job details
  3. Generate tailored resume content using the fine-tuned GPT-2 model
"""

import json
import gradio as gr
from scrape_jobs import scrape_job, scrape_job_from_text
from resume_tailor import load_generator, generate_resume_section


generator = None


def get_generator():
    global generator
    if generator is None:
        generator = load_generator()
    return generator


def scrape_from_url(url: str):
    if not url.strip():
        return "Please enter a job posting URL.", ""
    try:
        job = scrape_job(url.strip())
        return json.dumps(job, indent=2), ""
    except Exception as e:
        return f"Error scraping URL: {e}", ""


def scrape_from_text(job_text: str):
    if not job_text.strip():
        return "Please paste a job description.", ""
    try:
        job = scrape_job_from_text(job_text.strip())
        return json.dumps(job, indent=2), ""
    except Exception as e:
        return f"Error extracting job details: {e}", ""


def generate_from_job_json(job_json: str, max_length: int, temperature: float):
    if not job_json.strip():
        return "Scrape a job first, then click Generate."
    try:
        job = json.loads(job_json)
    except json.JSONDecodeError:
        return "Invalid job JSON. Use the Scrape tab first."
    try:
        gen = get_generator()
        return generate_resume_section(
            job,
            max_length=int(max_length),
            temperature=temperature,
            generator=gen,
        )
    except Exception as e:
        return f"Error generating resume content: {e}"


with gr.Blocks(title="Resume AI + ScrapeGraphAI") as demo:
    gr.Markdown(
        """
        # Resume AI — Powered by GPT-2 + LoRA + ScrapeGraphAI

        **Step 1:** Scrape a job description (URL or paste text)
        **Step 2:** Generate tailored resume content with the fine-tuned model
        """
    )

    job_json_state = gr.State("")

    with gr.Tab("Step 1a — Scrape from URL"):
        url_input = gr.Textbox(
            label="Job Posting URL",
            placeholder="https://jobs.example.com/devops-engineer-123",
        )
        scrape_url_btn = gr.Button("Scrape Job")
        url_job_output = gr.Code(label="Extracted Job Details (JSON)", language="json")

        scrape_url_btn.click(
            fn=scrape_from_url,
            inputs=[url_input],
            outputs=[url_job_output, job_json_state],
        )
        url_job_output.change(lambda x: x, inputs=url_job_output, outputs=job_json_state)

    with gr.Tab("Step 1b — Paste Job Description"):
        text_input = gr.Textbox(
            label="Paste Job Description Text",
            lines=10,
            placeholder="Copy and paste the full job description here...",
        )
        scrape_text_btn = gr.Button("Extract Job Details")
        text_job_output = gr.Code(label="Extracted Job Details (JSON)", language="json")

        scrape_text_btn.click(
            fn=scrape_from_text,
            inputs=[text_input],
            outputs=[text_job_output, job_json_state],
        )
        text_job_output.change(lambda x: x, inputs=text_job_output, outputs=job_json_state)

    with gr.Tab("Step 2 — Generate Resume Content"):
        gr.Markdown("Paste the JSON from Step 1, or it auto-fills after scraping.")
        job_json_input = gr.Code(
            label="Job Details JSON",
            language="json",
            lines=12,
        )
        with gr.Row():
            max_length_slider = gr.Slider(
                50, 500, value=300, step=10, label="Max Length (tokens)"
            )
            temperature_slider = gr.Slider(
                0.1, 1.5, value=0.8, step=0.05, label="Creativity (temperature)"
            )
        generate_btn = gr.Button("Generate Resume Content", variant="primary")
        resume_output = gr.Textbox(label="Generated Resume Content", lines=12)

        generate_btn.click(
            fn=generate_from_job_json,
            inputs=[job_json_input, max_length_slider, temperature_slider],
            outputs=[resume_output],
        )

    # Keep JSON in sync between tabs
    job_json_state.change(lambda x: x, inputs=job_json_state, outputs=job_json_input)


if __name__ == "__main__":
    demo.launch()
