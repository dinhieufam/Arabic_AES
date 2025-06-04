# -*- coding: utf-8 -*-

import pandas as pd
import torch
import json
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "inceptionai/jais-family-13b-chat"

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


prompt_eng = "### Instruction: You are a helpful assistant. Complete the conversation between [|Human|] and [|AI|]:\n### Input: [|Human|] {Question}\n[|AI|]\n### Response :"
# prompt_ar = "### Instruction:اسمك \"جيس\" وسميت على اسم جبل جيس اعلى جبل في الامارات. تم بنائك بواسطة Inception في الإمارات. أنت مساعد مفيد ومحترم وصادق. أجب دائمًا بأكبر قدر ممكن من المساعدة، مع الحفاظ على البقاء أمناً. أكمل المحادثة بين [|Human|] و[|AI|] :\n### Input:[|Human|] {Question}\n[|AI|]\n### Response :"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)


def get_response(text, tokenizer=tokenizer, model=model):
    # Tokenize with padding and truncation, and move to device
    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=1536  # Set explicit max length
    ).to(device)

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Generate model output with attention mask
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512
        )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids, outputs)
    ]

    decoded_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    output = decoded_output[0]
    response = output.split("### Response :")[-1]
    return response


df = pd.read_excel("../dataset.xlsx")
results = []

for essay_id, essay_text in zip(df['essay_id'], df['text']):
    print("😍 Processing the essay with ID: ", essay_id)
    row = {"essay_id": essay_id}
    total = 0
    for rubric in RUBRICS:
        # Run the model and parse the response
        prompt = prompt_eng.format_map({'Question': build_prompt(rubric, essay_text)})
        output = get_response(prompt)

        print("🔍 Output:\n", output)

        json_data = {}
        try:
            # json_data = json.loads(output.split("{", 1)[1].rsplit("}", 1)[0].join(["{", "}"]))
            score_match = output.split('"score": ')[1].split(',')[0]
            # print(f"Score Match:\n {score_match}")
            json_data["score"] = int(min(float(score_match), 5)) if rubric != "relevance" else int(min(float(score_match), 2))
            # print(f"JSON Data:\n {json_data}")
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
output_file = "../predictions/model_4/prompt_3.csv"
fieldnames = ["essay_id", "organization", "vocabulary", "style", "development", 
              "mechanics", "structure", "relevance", "final_score", "total_score"]

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"💾 Saved results to {output_file}")