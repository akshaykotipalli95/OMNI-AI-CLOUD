"""
FLAN-T5 Large Summarization (Improved Version)
Stable summarization without repetition loops
"""

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# =====================================================
# 1️⃣ DEVICE CONFIGURATION (GPU IF AVAILABLE)
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🚀 Using device: {device}")

# =====================================================
# 2️⃣ LOAD MODEL FROM LOCAL PATH
# =====================================================

MODEL_PATH = r"C:\Users\kotip\Desktop\OmniAI-Cloud\backend\models\downloaded_models\flan-t5-large"

print("\n⏳ Loading FLAN-T5 Large model...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

print("✅ Model Loaded Successfully!\n")

# =====================================================
# 3️⃣ LARGE SAMPLE TEXT
# =====================================================

text = """
Artificial Intelligence (AI) has rapidly evolved over the past decade and is now
a fundamental technology shaping industries across the globe. AI systems are capable
of processing enormous amounts of structured and unstructured data, identifying patterns,
and making intelligent decisions with minimal human intervention.

Machine learning, a subset of AI, allows systems to learn from historical data
and improve performance over time. Deep learning, which uses neural networks with
multiple layers, has significantly advanced fields such as computer vision,
natural language processing, and speech recognition.

In healthcare, AI assists doctors in diagnosing diseases using medical imaging
analysis and predictive analytics. In finance, AI is used for fraud detection,
algorithmic trading, and risk assessment. The transportation sector benefits from
AI-powered autonomous vehicles that rely on sensors, computer vision, and
real-time decision-making systems.

Despite its advantages, AI raises ethical and societal concerns. Issues such as
job displacement, algorithmic bias, privacy invasion, and lack of transparency
are widely debated. Governments and research institutions are working on
regulatory frameworks to ensure responsible AI deployment.

Looking ahead, AI is expected to play a critical role in solving global challenges,
including climate change, sustainable energy optimization, smart city development,
and personalized education systems.
"""

# =====================================================
# 4️⃣ STRONG INSTRUCTION PROMPT
# =====================================================

prompt = f"""
You are an expert academic writer.

Summarize the following article in ONE clear and concise paragraph.
Do not repeat phrases.
Do not copy sentences exactly.
Write in formal academic tone.

Article:
{text}

Summary:
"""

# Tokenize
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    max_length=1024,
    truncation=True
).to(device)

# =====================================================
# 5️⃣ GENERATE SUMMARY (FIXED SETTINGS)
# =====================================================

print("🤖 Generating summary...\n")

with torch.no_grad():
    summary_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=120,          # safer than max_length
        min_new_tokens=40,
        do_sample=True,              # prevents beam loop repetition
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=2.5,      # VERY IMPORTANT
        no_repeat_ngram_size=3,      # prevents repeating phrases
        length_penalty=1.5,
        early_stopping=True
    )

summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# =====================================================
# 6️⃣ PRINT RESULT
# =====================================================

print("=" * 70)
print("📄 ORIGINAL TEXT:\n")
print(text.strip())

print("\n" + "=" * 70)
print("✅ SUMMARY:\n")
print(summary.strip())
print("=" * 70)
