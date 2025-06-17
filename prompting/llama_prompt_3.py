# -*- coding: utf-8 -*-

import pandas as pd
import torch
import json
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Step 1: Define your model path (replace this with actual path or Hugging Face repo) ===
model_path = "meta-llama/Llama-2-7b-chat-hf"

# === Step 2: Rubric guides and prompt builder ===
RUBRICS = ["organization", "vocabulary", "style", "development", "mechanics", "structure", "relevance"]

RUBRIC_GUIDES = {
    "organization": {
        "arabic": "التنظيم",
        "guide": "1. هل المقال منظم جيدًا؟\n2. هل هناك مقدمة وجسم وخاتمة واضحة؟\n3. هل تسلسل الفقرات منطقي؟",
        "scoring": "0-5"
    },
    "vocabulary": {
        "arabic": "المفردات",
        "guide": "1. ما مدى تنوع المفردات؟\n2. هل يوجد تكرار أو استخدام غير مناسب؟\n3. هل المفردات دقيقة وتخدم المعنى؟",
        "scoring": "0-5"
    },
    "style": {
        "arabic": "الأسلوب",
        "guide": "1. هل الأسلوب ملائم وشيق؟\n2. هل توجد سلاسة في التعبير؟\n3. هل التراكيب والأساليب مناسبة؟",
        "scoring": "0-5"
    },
    "development": {
        "arabic": "تطوير المحتوى",
        "guide": "1. هل تم دعم الأفكار بأمثلة؟\n2. هل هناك تفصيل كافٍ؟\n3. هل الحجة مقنعة؟",
        "scoring": "0-5"
    },
    "mechanics": {
        "arabic": "الميكانيكا اللغوية",
        "guide": "1. هل توجد أخطاء نحوية أو إملائية؟\n2. هل علامات الترقيم مستخدمة بشكل صحيح؟",
        "scoring": "0-5"
    },
    "structure": {
        "arabic": "التراكيب النحوية",
        "guide": "1. هل الجمل سليمة؟\n2. هل التركيب النحوي واضح ومنضبط؟",
        "scoring": "0-5"
    },
    "relevance": {
        "arabic": "مدى الصلة بالموضوع",
        "guide": "1. هل المقال يعالج موضوع التواصل والتكنولوجيا؟\n2. هل تنسجم الفكرة العامة مع الموضوع؟",
        "scoring": "0-2"
    }
}

def build_prompt(rubric, essay_text):
    rubric_info = RUBRIC_GUIDES[rubric]

    with open(f'rubric_examples/{rubric}.txt', 'r', encoding='utf-8') as f:
        example = f.read()

    return f"""
أنت مقيم لغوي مختص في تقييم مهارة [{rubric_info['arabic']}] في المقالات المكتوبة باللغة العربية.

ستقوم بتقييم المقال بناءً على هذه المهارة فقط.

يرجى اتباع الخطوات التالية:
1. اقرأ المقال جيداً.
2. اتبع دليل التقييم التالي:
{rubric_info['guide']}
3. حدد درجة من {rubric_info['scoring']}.
4. قدم مبررات واضحة لقرارك.

يتضمن المعيار أدناه:
- وصفًا لما يقيسه
- ثلاثة مستويات كأمثلة (الدرجات: ١، ٣، ٥)
\"\"\"
{example}
\"\"\"
🎯 استخدم هذه الأمثلة لمقارنة المقال الذي تقوم بتقييمه وتبرير النتيجة وفقًا لذلك.

✏️ المقال:
\"\"\"
{essay_text}
\"\"\"

أجب بصيغة JSON فقط بهذا الشكل:
{{
    "score": X,
    "justification": "..."
}}
"""

# === Step 3: Load the model ===
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

# === Step 4: Define response function ===
def get_response(prompt, tokenizer=tokenizer, model=model):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.95,
        temperature=0.7
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "{" in decoded and "}" in decoded:
        return decoded.split("{", 1)[1].rsplit("}", 1)[0].join(["{", "}"])
    return decoded

# === Step 5: Load essay dataset ===
df = pd.read_excel("dataset.xlsx")
results = []

for essay_id, essay_text in zip(df['essay_id'], df['text']):
    print(f"📝 Essay ID: {essay_id}")
    row = {"essay_id": essay_id}
    total = 0
    for rubric in RUBRICS:
        prompt = build_prompt(rubric, essay_text)
        output = get_response(prompt)

        print(f"📤 LLaMA output:\n{output}")
        try:
            data = json.loads(output)
            score = int(min(float(data["score"]), 5)) if rubric != "relevance" else int(min(float(data["score"]), 2))
        except Exception as e:
            print("⚠️ JSON parse failed:", e)
            score = 0

        row[rubric] = score
        total += score
        print(f"✅ {rubric}: {score}")

    row["final_score"] = total
    row["total_score"] = total
    results.append(row)

# === Step 6: Save results ===
output_file = "predictions/model_6/prompt_3.csv"
fieldnames = ["essay_id"] + RUBRICS + ["final_score", "total_score"]

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"✅ Results saved to: {output_file}")
