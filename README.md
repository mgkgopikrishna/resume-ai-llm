# 🤖 Resume AI — Custom LLM Built From Scratch

> **A real AI brain trained to read and write resumes — built by a DevOps Engineer using zero dollars and free cloud GPUs**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=flat-square&logo=gradio&logoColor=white)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=flat-square)

---

## 🧒 What Is This? (Explain Like I Am 10 Years Old)

Imagine you want to teach your dog a new trick. You show the dog the trick 1000 times. After a lot of practice, the dog learns it!

This project does the same thing — but instead of a dog, it is a computer program called an **AI model**. And instead of tricks, we taught it to **read and write professional resumes**.

We fed the AI **40 real resumes** from DevOps and Cloud engineers. The AI read them over and over again until it learned the patterns — things like how to describe job experience, what skills to list, how to write a summary.

Now the AI can **generate new resume content** all by itself — like a writing assistant that understands tech careers.

---

## 🎯 What Does This Project Do?

- 📄 **Reads 40 professional tech resumes** as training data
- 🧠 **Trains a GPT-2 language model from scratch** — meaning we built the AI brain ourselves
- 🔧 **Fine-tunes it with LoRA** — a smart trick to make training faster and cheaper
- 🚀 **Deploys it live on HuggingFace** — anyone in the world can use it for free
- 💬 **Chat with it using Gradio** — a simple web interface

---

## 🗺️ The Complete Journey — Step By Step

### Step 1 — Understanding What an LLM Is

**LLM** stands for **Large Language Model**. It is the technology behind ChatGPT, Claude, and Gemini.

Think of it like this:
- A normal calculator can add numbers
- An LLM can read and write human language

How does it work?
1. You give it millions of words to read
2. It learns the patterns — which words usually follow which words
3. After learning, it can predict and generate new text

Our model is small (GPT-2 with 117 million parameters) but the concept is exactly the same as the big models.

---

### Step 2 — Collecting the Training Data

**What is training data?**

Training data is the information we use to teach the AI. It is like the textbooks a student studies from.

For this project:
- We collected **40 professional resumes** from real DevOps, MLOps, and Cloud engineers
- Each resume had sections like: Summary, Work Experience, Skills, Certifications
- We formatted everything as clean plain text
- We saved it all in one file called `resume_data.txt`
- Total size: about **100 KB** of text

**Why resumes?**
Because we want the AI to understand the language of tech careers — the words, phrases, and structure that professional engineers use.

```
Example of what training data looks like:

SUMMARY
Results-driven DevOps Engineer with 5 years of experience building
CI/CD pipelines and managing Kubernetes clusters on AWS and Azure.

EXPERIENCE
Senior DevOps Engineer — Tesla (2024-2025)
- Built hybrid EKS/AKS clusters handling 500+ microservices
- Reduced deployment time by 60% using ArgoCD and GitOps
- Implemented MLflow and Kubeflow for ML pipeline automation
```

---

### Step 3 — Setting Up the Free GPU Environment

**What is a GPU?**

A GPU (Graphics Processing Unit) is a special chip that can do millions of math calculations at the same time. Training AI models needs massive amounts of math, so GPUs are essential.

The problem? GPUs are expensive. A good one costs thousands of dollars.

**The solution? Kaggle gives free GPUs!**

- Created a free account at **kaggle.com**
- Kaggle gives you **2x NVIDIA T4 GPUs for free**
- You get **30 hours of GPU time per week**
- This is more than enough to train our model

**Setting up the environment:**

```bash
# Install required libraries
pip install torch transformers datasets tiktoken

# Clone nanoGPT — the training framework by Andrej Karpathy
git clone https://github.com/karpathy/nanoGPT
cd nanoGPT
```

**What is nanoGPT?**
It is a simplified, clean implementation of GPT written by Andrej Karpathy (ex-Tesla AI Director). It is like a starter kit for building your own language model.

---

### Step 4 — Preparing the Data (Tokenization)

**What is tokenization?**

AI models cannot read words directly. They read **numbers**. Tokenization converts words into numbers.

Think of it like a secret code:
- "Hello" → 15496
- "DevOps" → 6216
- "Engineer" → 10616

**The tokenizer we used: tiktoken (GPT-2 BPE)**

BPE stands for **Byte Pair Encoding**. It is a clever system that breaks text into small chunks called tokens. A token is roughly 3/4 of a word.

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

**Why split into train and validation?**
- Training data (90%): The AI learns from this
- Validation data (10%): We test if the AI actually understood or just memorized

---

### Step 5 — Training the Model From Scratch

**What does "training from scratch" mean?**

Most people download a pre-trained model (like GPT-2) that was already trained by OpenAI on billions of web pages.

Training from scratch means we started with a model that knows **absolutely nothing** — like a newborn baby. Then we taught it using only our resume data.

**The model architecture: Transformer**

A transformer is the core technology behind all modern AI language models. Here is how it works in simple terms:

```
Input: "DevOps Engineer with experience in"
         ↓
Tokenize: [6216, 10616, 351, 1998, 287]
         ↓
Embedding Layer: Convert each token number into a list of 768 numbers
         ↓
Self-Attention (12 layers): Each word looks at all other words
                             to understand context
         ↓
Feed-Forward Network: Process the attended information
         ↓
Output: Predict the next token = "Kubernetes"
```

**Our model configuration:**

```python
# config/train_resume.py

# Model size
n_layer = 6        # 6 transformer layers (GPT-2 has 12)
n_head = 6         # 6 attention heads
n_embd = 384       # 384 dimensions (GPT-2 has 768)
block_size = 256   # Read 256 tokens at a time

# Training settings
batch_size = 64           # Process 64 examples at once
learning_rate = 1e-3      # How fast the model learns
max_iters = 5000          # Train for 5000 steps
eval_interval = 500       # Check progress every 500 steps

# Output
out_dir = "out-resume"    # Save model here
```

**Running the training:**

```bash
python train.py config/train_resume.py
```

**What happens during training?**

```
Iteration 0:    loss = 4.26  ← Model knows nothing, guessing randomly
Iteration 500:  loss = 3.14  ← Starting to learn patterns
Iteration 1000: loss = 2.67  ← Learning resume structure
Iteration 2000: loss = 2.01  ← Understanding tech terminology
Iteration 3000: loss = 1.72  ← Getting good at resume language
Iteration 5000: loss = 1.42  ← Final model — much better!
```

**What is loss?**
Loss measures how wrong the model is. Lower loss = better model. We started at 4.26 and ended at 1.42 — that is a huge improvement!

Training time: ~45 minutes on Kaggle's free T4 GPU

---

### Step 6 — Fine-Tuning With LoRA

**What is fine-tuning?**

After training from scratch we have a basic model. Fine-tuning makes it smarter for a specific task.

Think of it like this:
- Training from scratch = teaching someone to speak English
- Fine-tuning = teaching that English speaker to write business emails specifically

**What is LoRA?**

LoRA stands for **Low-Rank Adaptation**. It is a genius technique that makes fine-tuning 98% cheaper.

Normal fine-tuning: Update ALL 117 million parameters → very expensive
LoRA: Add tiny "adapter" layers and only train THOSE → only 2 million parameters!

```
GPT-2 Model (117M parameters) — FROZEN, not changed
        ↓
LoRA Adapters (2M parameters) — TRAINED on our resume data
        ↓
Final model = GPT-2 brain + Resume expertise adapters
```

**The fine-tuning code:**

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import get_peft_model, LoraConfig, TaskType

# Load base GPT-2
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Configure LoRA adapters
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,              # Rank — controls adapter size
    lora_alpha=32,     # Scaling factor
    lora_dropout=0.1,  # Prevent overfitting
    target_modules=["c_attn", "c_proj"]  # Which layers to adapt
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# See how many parameters we are actually training
model.print_trainable_parameters()
# Output: trainable params: 2,097,152 || all params: 124,734,720 || trainable%: 1.68%
```

**Creating instruction-response training pairs:**

```
Example training pair:

INSTRUCTION: Write a professional summary for a DevOps Engineer with 5 years of experience

RESPONSE: Results-driven DevOps Engineer with 5 years of experience
specializing in cloud-native infrastructure on AWS, Azure, and GCP.
Proven track record of building CI/CD pipelines, managing Kubernetes
clusters, and implementing MLOps workflows that reduced deployment
time by 40% and improved system reliability by 35%.
```

---

### Step 7 — Deploying to HuggingFace

**What is HuggingFace?**

HuggingFace is like GitHub but for AI models. You can upload your model and share it with the world for free.

**Uploading the model:**

```python
from huggingface_hub import HfApi

api = HfApi()

# Upload model to HuggingFace Hub
api.upload_folder(
    folder_path="./resume-ai-model",
    repo_id="GopiKrishna88/resume-ai-llm",
    repo_type="model"
)

print("Model uploaded successfully!")
```

**Building the Gradio demo:**

```python
import gradio as gr
from transformers import pipeline

# Load our trained model
generator = pipeline(
    "text-generation",
    model="GopiKrishna88/resume-ai-llm"
)

def generate_resume_content(prompt, max_length=200):
    """Generate resume content based on user prompt"""
    result = generator(
        prompt,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.8,    # Creativity level (0=boring, 1=creative)
        top_p=0.9,          # Only consider top 90% probable words
        do_sample=True
    )
    return result[0]["generated_text"]

# Create the web interface
demo = gr.Interface(
    fn=generate_resume_content,
    inputs=[
        gr.Textbox(
            label="Enter your prompt",
            placeholder="Write a DevOps Engineer summary with Kubernetes experience..."
        ),
        gr.Slider(50, 500, value=200, label="Response Length")
    ],
    outputs=gr.Textbox(label="Generated Resume Content"),
    title="🤖 Resume AI — Powered by GPT-2 + LoRA",
    description="Enter a prompt and the AI will generate professional resume content"
)

demo.launch()
```

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Training Loss (Start) | 4.26 |
| Training Loss (End) | 1.42 |
| Parameters Trained (LoRA) | 2M out of 117M |
| Training Cost | $0 (free Kaggle GPU) |
| Training Time | ~45 minutes |
| Deployment | Live on HuggingFace |

---

## 🛠️ Full Tech Stack

| Tool | What It Does |
|------|-------------|
| **Python** | The programming language everything is written in |
| **PyTorch** | The deep learning framework (like Lego blocks for AI) |
| **nanoGPT** | The training framework to build GPT from scratch |
| **tiktoken** | Converts words to numbers (tokenization) |
| **HuggingFace Transformers** | Library to work with pre-trained models |
| **PEFT / LoRA** | Makes fine-tuning cheap and fast |
| **Gradio** | Creates the web demo interface |
| **Kaggle** | Free GPU cloud environment |
| **HuggingFace Hub** | Hosts and shares the model |

---

## 🔧 How To Run This Yourself

```bash
# 1. Clone the repository
git clone https://github.com/mgkgopikrishna/resume-ai-llm
cd resume-ai-llm

# 2. Install dependencies
pip install torch transformers datasets tiktoken peft gradio huggingface_hub

# 3. Prepare your training data
# Add your text data to data/resume_data.txt

# 4. Tokenize the data
python data/prepare.py

# 5. Train from scratch
python train.py config/train_resume.py

# 6. Fine-tune with LoRA
python finetune_lora.py

# 7. Run the demo locally
python app.py
# Open http://localhost:7860 in your browser
```

---

## 💡 Key Lessons Learned

1. **You do not need millions of dollars to build AI** — free tools and creativity are enough
2. **LoRA is a game changer** — 98% fewer parameters to train means anyone can fine-tune
3. **Quality data beats quantity** — 40 well-formatted resumes taught more than 4000 random ones would
4. **Loss is your compass** — watch it fall during training to know you are on the right path
5. **HuggingFace democratizes AI** — sharing your model with the world takes 5 minutes

---

## 👨‍💻 Built By

**Gopi Krishna Marka** — MLOps and DevOps Engineer learning AI from scratch

*Proof that a DevOps Engineer can build and deploy a real LLM using zero dollars*

[![HuggingFace Demo](https://img.shields.io/badge/🤗_Live_Demo-HuggingFace-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/GopiKrishna88/resume-ai-demo)
