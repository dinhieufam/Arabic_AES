import openai
import time
import pandas as pd
import json
import csv
from secret_key import openai_key

openai.api_key = openai_key

output_file = "predictions/gpt4/prompt_level_3.csv"

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

def build_system_prompt(rubric, max_tokens=7000):
    info = RUBRIC_GUIDES[rubric]
    with open(f"rubric_examples/{rubric}.txt", encoding='utf-8') as f:
        example = f.read()

    # Truncate rubric example if too long
    if len(example) > 3000:
        print(f"⚠️ Truncating example for rubric: {rubric}")
        example = example[:3000] + "\n...[truncated]"

    return f"""أنت مقيم لغوي مختص في تقييم مهارة [{info['arabic']}] في المقالات المكتوبة باللغة العربية.

ستقوم بتقييم المقال بناءً على هذه المهارة فقط.

يرجى اتباع الخطوات التالية:
1. اقرأ المقال جيداً.
2. اتبع دليل التقييم التالي:
{info['guide']}
3. حدد درجة من {info['scoring']}.
4. قدم مبررات واضحة لقرارك.

يتضمن المعيار أدناه:
- وصفًا لما يقيسه
- ثلاثة مستويات كأمثلة (الدرجات: ١، ٣، ٥)

\"\"\"
{example}
\"\"\"

🎯 استخدم هذه الأمثلة لمقارنة المقال الذي تقوم بتقييمه وتبرير النتيجة وفقًا لذلك.

أجب بصيغة JSON فقط بهذا الشكل:
{{
    "score": X,
    "justification": "..."
}}
"""

def build_user_prompt(essay_text, max_chars=2000):
    # Truncate long essay
    if len(essay_text) > max_chars:
        print("⚠️ Truncating essay text")
        essay_text = essay_text[:max_chars] + "\n...[truncated]"

    return f"""✏️ المقال:
\"\"\"
{essay_text}
\"\"\"
"""


def get_gpt4_response(system_prompt, user_prompt):
    while True:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(10)

df = pd.read_excel("dataset.xlsx")
results = []

for essay_id, essay_text in zip(df['essay_id'], df['text']):
    print(f"📝 Scoring Essay ID: {essay_id}")
    row = {"essay_id": essay_id}
    total = 0

    for rubric in RUBRICS:
        system_prompt = build_system_prompt(rubric)
        user_prompt = build_user_prompt(essay_text)

        output = get_gpt4_response(system_prompt, user_prompt)
        print(f"📜 GPT Output for {rubric}:\n{output}")

        try:
            json_start = output.find('{')
            json_end = output.rfind('}') + 1
            json_str = output[json_start:json_end]
            parsed = json.loads(json_str)

            score = float(parsed.get("score", 0))
            score = int(min(score, 2 if rubric == "relevance" else 5))
        except Exception as e:
            print(f"⚠️ Parsing error for {rubric}: {e}")
            score = 0

        row[rubric] = score
        total += score

    row["final_score"] = round(total)
    row["total_score"] = round(total)

    results.append(row)
    time.sleep(15)

fieldnames = ["essay_id"] + RUBRICS + ["final_score", "total_score"]

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"💾 Saved GPT-4 results to {output_file}")
