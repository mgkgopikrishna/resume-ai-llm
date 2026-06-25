# 🤖 Resume AI — Custom LLM Built From Scratch

> **A real AI brain trained to read and write resumes — built by a DevOps Engineer using zero dollars and free cloud GPUs**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=flat-square&logo=gradio&logoColor=white)
![ScrapeGraphAI](https://img.shields.io/badge/ScrapeGraphAI-4B8BBE?style=flat-square&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=flat-square)

---

## 🎯 What Does This Project Do?
- 📄 Reads 40 professional tech resumes as training data
- 🧠 Trains a GPT-2 language model from scratch — meaning we built the AI brain ourselves
- 🔧 Fine-tunes it with LoRA — a smart trick to make training faster and cheaper
- 🚀 Deploys it live on HuggingFace — anyone in the world can use it for free
- 💬 Chat with it using Gradio — a simple web interface

## 🗺️ The Complete Journey — Step By Step

### Step 1 — Understanding What an LLM Is

LLM stands for Large Language Model. It is the technology behind ChatGPT, Claude, and Gemini.

How does it work?
- You give it millions of words to read
- It learns the patterns — which words usually follow which words
- After learning, it can predict and generate new text

Our model is small (GPT-2 with 117 million parameters) but the concept is exactly the same as the big models.

### Step 2 — Collecting the Training Data

Training data is the information we use to teach the AI.

For this project:
- We collected 40 professional resumes from real DevOps, MLOps, and Cloud engineers
- Each resume had sections like: Summary, Work Experience, Skills, Certifications
- We formatted everything as clean plain text
- We saved it all in one file called `resume_data.txt`
- Total size: about 100 KB of text

Example of what training data looks like:

```
SUMMARY
Results-driven DevOps Engineer with 5 years of experience building
CI/CD pipelines and managing Kubernetes clusters on AWS and Azure.

EXPERIENCE
Senior DevOps Engineer — Tesla (2024-2025)
- Built hybrid EKS/AKS clusters handling 500+ microservices
- Reduced deployment time by 60% using ArgoCD and GitOps
- Implemented MLflow and Kubeflow for ML pipeline automation
```

### Step 3 — Setting Up the Free GPU Environment

Kaggle gives free GPUs:
- Created a free account at kaggle.com
- Kaggle gives you 2x NVIDIA T4 GPUs for free
- You get 30 hours of GPU time per week

Setting up the environment:

```bash
# Install required libraries
pip install torch transformers datasets tiktoken

# Clone nanoGPT — the training framework by Andrej Karpathy
git clone https://github.com/karpathy/nanoGPT
cd nanoGPT
```

### Step 4 — Preparing the Data (Tokenization)

Tokenization converts words into numbers that the model can process. We used tiktoken (GPT-2 BPE).

```python
import tiktoken
import numpy as np

# Load GPT-2's tokenizer
encoder = tiktoken.get_encoding("gpt2")

# Read our resume training data
with open("resume_data.txt", "r") as f:
    text = f.read()

# Convert all words to numbers (tokens)
tokens = encoder.encode_ordinary(text)
tokens = np.array(tokens, dtype=np.uint16)

# Split into training (90%) and validation (10%)
split = int(0.9 * len(tokens))
train_data = tokens[:split]
val_data = tokens[split:]

# Save as binary files for fast loading
train_data.tofile("data/train.bin")
val_data.tofile("data/val.bin")

print(f"Total tokens: {len(tokens):,}")
print(f"Training tokens: {len(train_data):,}")
print(f"Validation tokens: {len(val_data):,}")
```

### Step 5 — Training the Model From Scratch

Our model configuration:

```python
# config/train_resume.py

# Model size
n_layer = 6        # 6 transformer layers (GPT-2 has 12)
n_head = 6         # 6 attention heads
n_embd = 384       # 384 dimensions (GPT-2 has 768)
block_size = 256   # Read 256 tokens at a time

# Training settings
batch_size = 64        # Process 64 examples at once
learning_rate = 1e-3   # How fast the model learns
max_iters = 5000       # Train for 5000 steps
eval_interval = 500    # Check progress every 500 steps

# Output
out_dir = "out-resume"  # Save model here
```

Running the training:

```bash
python train.py config/train_resume.py
```

Training progress:

```
Iteration    0: loss = 4.26  -- Model knows nothing, guessing randomly
Iteration  500: loss = 3.14  -- Starting to learn patterns
Iteration 1000: loss = 2.67  -- Learning resume structure
Iteration 2000: loss = 2.01  -- Understanding tech terminology
Iteration 3000: loss = 1.72  -- Getting good at resume language
Iteration 5000: loss = 1.42  -- Final model — much better!
```

Training time: ~45 minutes on Kaggle's free T4 GPU

### Step 6 — Fine-Tuning With LoRA

LoRA (Low-Rank Adaptation) makes fine-tuning 98% cheaper by only training small adapter layers instead of the full model.

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import get_peft_model, LoraConfig, TaskType

# Load base GPT-2
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Configure LoRA adapters
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                          # Rank
    lora_alpha=32,                 # Scaling factor
    lora_dropout=0.1,              # Prevent overfitting
    target_modules=["c_attn", "c_proj"]
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 2,097,152 || all params: 124,734,720 || trainable%: 1.68%
```

### Step 7 — Deploying to HuggingFace

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./resume-ai-model",
    repo_id="GopiKrishna88/resume-ai-llm",
    repo_type="model"
)
print("Model uploaded successfully!")
```

Building the Gradio demo:

```python
import gradio as gr
from transformers import pipeline

generator = pipeline("text-generation", model="GopiKrishna88/resume-ai-llm")

def generate_resume_content(prompt, max_length=200):
    result = generator(prompt, max_length=max_length, num_return_sequences=1,
                       temperature=0.8, top_p=0.9, do_sample=True)
    return result[0]["generated_text"]

demo = gr.Interface(
    fn=generate_resume_content,
    inputs=[
        gr.Textbox(label="Enter your prompt",
                   placeholder="Write a DevOps Engineer summary with Kubernetes experience..."),
        gr.Slider(50, 500, value=200, label="Response Length")
    ],
    outputs=gr.Textbox(label="Generated Resume Content"),
    title="Resume AI — Powered by GPT-2 + LoRA"
)
demo.launch()
```

## 📊 Results

| Metric | Value |
|--------|-------|
| Training Loss (Start) | 4.26 |
| Training Loss (End) | 1.42 |
| Parameters Trained (LoRA) | 2M out of 117M |
| Training Cost | $0 (free Kaggle GPU) |
| Training Time | ~45 minutes |
| Deployment | Live on HuggingFace |

## 🔍 ScrapeGraphAI Integration — Job Description Scraping

This project now uses [ScrapeGraphAI](https://github.com/ScrapeGraphAI/Scrapegraph-ai) to automatically extract structured data from job postings. You only need to paste a URL (or raw text) and the AI reads the page, pulls out the skills, responsibilities, and requirements, and feeds them directly into the resume generator.

### How It Works

```
Job Posting URL
      │
      ▼
 ScrapeGraphAI (SmartScraperGraph)
      │  uses LLM to parse the page
      ▼
 Structured JSON
 {
   "job_title": "Senior DevOps Engineer",
   "required_skills": ["Kubernetes", "Terraform", "AWS"],
   "responsibilities": ["Build CI/CD pipelines", ...],
   ...
 }
      │
      ▼
 Resume AI (GPT-2 + LoRA)
      │  generates tailored content
      ▼
 Tailored Resume Section
```

### Scraping a Job from the Command Line

```bash
# Scrape a job posting URL directly
python scrape_jobs.py https://jobs.example.com/devops-engineer

# Output: structured JSON with title, skills, responsibilities, etc.
```

### LLM Options for Scraping (pick one)

| Provider | Setup | Cost |
|----------|-------|------|
| Ollama (local) | `ollama pull llama3.2` | Free |
| Groq | `export GROQ_API_KEY=your_key` | Free tier |
| OpenAI | `export OPENAI_API_KEY=your_key` | Paid |

The scraper auto-detects which provider to use based on your environment variables, falling back to Ollama if none are set.

## 🛠️ Full Tech Stack

| Tool | What It Does |
|------|-------------|
| Python | Main programming language |
| PyTorch | Deep learning framework |
| nanoGPT | Training framework to build GPT from scratch |
| tiktoken | Converts words to numbers (tokenization) |
| HuggingFace Transformers | Library for pre-trained models |
| PEFT / LoRA | Makes fine-tuning cheap and fast |
| Gradio | Creates the web demo interface |
| Kaggle | Free GPU cloud environment |
| HuggingFace Hub | Hosts and shares the model |
| ScrapeGraphAI | AI-powered job description scraper |

## 🔧 How To Run This Yourself

```bash
# 1. Clone the repository
git clone https://github.com/mgkgopikrishna/resume-ai-llm
cd resume-ai-llm

# 2. Install dependencies
pip install -r requirements.txt
playwright install

# 3. (Optional) Set up a free LLM for job scraping
#    Option A: Install Ollama and pull a model
#      brew install ollama && ollama pull llama3.2
#    Option B: Get a free Groq API key at console.groq.com
#      export GROQ_API_KEY=your_key_here

# 4. Add your training data to data/resume_data.txt

# 5. Tokenize the data
python data/prepare.py

# 6. Train from scratch
python train.py config/train_resume.py

# 7. Fine-tune with LoRA
python finetune_lora.py

# 8. Run the full app (scraping + generation)
python app.py
# Open http://localhost:7860 in your browser
```

### Scraping Only (no training needed)

```bash
# Scrape any job posting URL
python scrape_jobs.py https://jobs.example.com/devops-engineer-123

# Or use it in Python
from scrape_jobs import scrape_job
job = scrape_job("https://jobs.example.com/devops-engineer-123")
print(job["required_skills"])

# Generate tailored resume content from the scraped job
from resume_tailor import load_generator, generate_resume_section
gen = load_generator()
content = generate_resume_section(job, generator=gen)
print(content)
```

## 💡 Key Lessons Learned

- You do not need millions of dollars to build AI — free tools and creativity are enough
- LoRA is a game changer — 98% fewer parameters to train means anyone can fine-tune
- Quality data beats quantity — 40 well-formatted resumes outperform 4000 random ones
- Loss is your compass — watch it fall during training to know you are on the right path
- HuggingFace democratizes AI — sharing your model with the world takes 5 minutes

## 👨‍💻 Built By

**Gopi Krishna Marka** — MLOps and DevOps Engineer learning AI from scratch

* can build and deploy a real LLM using zero dollars*
