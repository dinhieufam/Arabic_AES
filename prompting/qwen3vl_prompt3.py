# -*- coding: utf-8 -*-

import pandas as pd
import torch
import json
import csv
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

model_path = "Qwen/Qwen3-VL-8B-Instruct"

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

    with open(f'../rubric_examples/{rubric}.txt', 'r', encoding='utf-8') as f:
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
استخدم هذه الأمثلة لمقارنة المقال الذي تقوم بتقييمه وتبرير النتيجة وفقًا لذلك.

المقال:
\"\"\"
{essay_text}
\"\"\"

أجب بصيغة JSON فقط بهذا الشكل:
{{
    "score": X,
    "justification": "..."
}}
"""

prompt_eng = "### Instruction: You are a helpful assistant. Complete the conversation between [|Human|] and [|AI|]:\n### Input: [|Human|] {Question}\n[|AI|]\n### Response :"

# Load model and processor
model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(model_path)

def get_response(text, processor=processor, model=model):
    # Prepare messages for the model
    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": text}
            ]
        }
    ]
    
    # Apply chat template and prepare inputs
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        
    # Process generated output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    return output_text

df = pd.read_excel("../dataset.xlsx")
results = []

for essay_id, essay_text in zip(df['essay_id'], df['text']):
    print("Processing the essay with ID: ", essay_id)
    row = {"essay_id": essay_id}
    total = 0
    for rubric in RUBRICS:
        # Run the model and parse the response
        prompt = prompt_eng.format_map({'Question': build_prompt(rubric, essay_text)})
        output = get_response(prompt)

        print("🔍 Output:\n", output)

        json_data = {}
        try:
            # Find JSON in response
            json_start = output.find('{')
            json_end = output.rfind('}') + 1
            json_str = output[json_start:json_end]
            
            # Parse scores
            json_data = json.loads(json_str)
            json_data["score"] = int(min(float(json_data["score"]), 5)) if rubric != "relevance" else int(min(float(json_data["score"]), 2))
            
        except Exception as e:
            json_data = {"score": 0, "justification": f"Parsing error: {str(e)}"}

        # Add the score to the row
        row[rubric] = json_data["score"]

        # Add the score to the total
        total += json_data["score"]

        print(f"📌 {rubric}: {json_data['score']}")

    row["final_score"] = round(total)
    row["total_score"] = round(total)

    print(row)

    results.append(row)

# Save results to CSV
output_file = "../predictions/qwen3vl/prompt_3.csv"
fieldnames = ["essay_id", "organization", "vocabulary", "style", "development", 
              "mechanics", "structure", "relevance", "final_score", "total_score"]

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"💾 Saved results to {output_file}")