import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def run_model_and_parse_response(rubric, essay_text, model, tokenizer):
    prompt = build_prompt(rubric, essay_text)

    # print(prompt)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize with padding and truncation, and move to device
    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding="longest",
        truncation=True
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

    # print(f"Raw Output:\n {output}")

    json_data = {}
    try:
        # json_data = json.loads(output.split("{", 1)[1].rsplit("}", 1)[0].join(["{", "}"]))
        score_match = output.split('"score": ')[1].split(',')[0]
        # print(f"Score Match:\n {score_match}")
        json_data["score"] = int(min(float(score_match), 5)) if rubric != "relevance" else int(min(float(score_match), 2))
        # print(f"JSON Data:\n {json_data}")
    except Exception as e:
        json_data = {"score": 0, "justification": f"Parsing error: {str(e)}"}

    return json_data
