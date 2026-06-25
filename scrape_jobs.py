"""
Scrape job descriptions from a URL using ScrapeGraphAI.

Requires either:
  - Ollama running locally: ollama pull llama3.2
  - A Groq API key set as GROQ_API_KEY env var (free tier available)
  - An OpenAI API key set as OPENAI_API_KEY env var
"""

import os
import json
from scrapegraphai.graphs import SmartScraperGraph


def build_llm_config() -> dict:
    """Pick an LLM provider based on available env vars, defaulting to Ollama."""
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if groq_key:
        return {"api_key": groq_key, "model": "groq/llama-3.1-8b-instant"}
    if openai_key:
        return {"api_key": openai_key, "model": "openai/gpt-4o-mini"}
    # Fall back to local Ollama (free, requires: ollama pull llama3.2)
    return {"model": "ollama/llama3.2", "model_tokens": 8192}


EXTRACT_PROMPT = """
Extract the following from this job posting and return as JSON:
{
  "job_title": "...",
  "company": "...",
  "location": "...",
  "summary": "2-3 sentence overview of the role",
  "required_skills": ["skill1", "skill2", ...],
  "preferred_skills": ["skill1", ...],
  "responsibilities": ["responsibility1", ...],
  "years_experience": "e.g. 3-5 years",
  "education": "e.g. Bachelor's in Computer Science"
}
"""


def scrape_job(url: str, verbose: bool = False) -> dict:
    """
    Scrape a job posting URL and return structured job details.

    Args:
        url: Public URL of the job posting.
        verbose: Print scraping progress if True.

    Returns:
        Dict with job title, company, skills, responsibilities, etc.
    """
    config = {
        "llm": build_llm_config(),
        "verbose": verbose,
        "headless": True,
    }

    scraper = SmartScraperGraph(
        prompt=EXTRACT_PROMPT,
        source=url,
        config=config,
    )

    result = scraper.run()
    return result


def scrape_job_from_text(job_text: str, verbose: bool = False) -> dict:
    """
    Extract structured job details from raw job description text.

    Args:
        job_text: Raw text of the job description (paste from job board).
        verbose: Print progress if True.

    Returns:
        Dict with job title, company, skills, responsibilities, etc.
    """
    config = {
        "llm": build_llm_config(),
        "verbose": verbose,
        "headless": True,
    }

    scraper = SmartScraperGraph(
        prompt=EXTRACT_PROMPT,
        source=job_text,
        config=config,
    )

    result = scraper.run()
    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python scrape_jobs.py <job-url>")
        print("Example: python scrape_jobs.py https://jobs.example.com/devops-engineer")
        sys.exit(1)

    url = sys.argv[1]
    print(f"Scraping: {url}\n")
    job = scrape_job(url, verbose=True)
    print(json.dumps(job, indent=2))
