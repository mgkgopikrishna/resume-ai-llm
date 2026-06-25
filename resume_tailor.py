"""
Generate tailored resume content using the fine-tuned GPT-2 model.

Given a structured job description (from scrape_jobs.py), this module
builds a focused prompt and uses the HuggingFace-hosted resume model
to generate role-specific resume sections.
"""

from transformers import pipeline


MODEL_ID = "GopiKrishna88/resume-ai-llm"


def load_generator(model_id: str = MODEL_ID):
    return pipeline(
        "text-generation",
        model=model_id,
        device_map="auto",
    )


def build_prompt(job: dict) -> str:
    """Convert structured job data into a resume-generation prompt."""
    title = job.get("job_title", "Engineer")
    skills = ", ".join(job.get("required_skills", [])[:6])
    responsibilities = job.get("responsibilities", [])
    top_resp = responsibilities[0] if responsibilities else ""
    years = job.get("years_experience", "")

    prompt = f"SUMMARY\n{title}"
    if years:
        prompt += f" with {years} of experience"
    if skills:
        prompt += f" skilled in {skills}"
    if top_resp:
        prompt += f". {top_resp}"
    prompt += ".\n\nEXPERIENCE\n"
    return prompt


def generate_resume_section(
    job: dict,
    max_length: int = 300,
    temperature: float = 0.8,
    top_p: float = 0.9,
    generator=None,
) -> str:
    """
    Generate a tailored resume section for the given job.

    Args:
        job: Structured job dict returned by scrape_jobs.scrape_job().
        max_length: Max tokens to generate.
        temperature: Sampling temperature (higher = more creative).
        top_p: Nucleus sampling probability.
        generator: Pre-loaded pipeline (created if not provided).

    Returns:
        Generated resume text as a string.
    """
    if generator is None:
        generator = load_generator()

    prompt = build_prompt(job)
    result = generator(
        prompt,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        num_return_sequences=1,
    )
    return result[0]["generated_text"]


if __name__ == "__main__":
    sample_job = {
        "job_title": "Senior DevOps Engineer",
        "required_skills": ["Kubernetes", "Terraform", "AWS", "CI/CD", "Docker"],
        "years_experience": "5+ years",
        "responsibilities": [
            "Design and maintain Kubernetes clusters on AWS EKS",
            "Build CI/CD pipelines using GitHub Actions and ArgoCD",
        ],
    }

    print("Loading model...")
    gen = load_generator()
    print("\nGenerating tailored resume content...\n")
    output = generate_resume_section(sample_job, generator=gen)
    print(output)
