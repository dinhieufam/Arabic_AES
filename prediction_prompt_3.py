
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
        "scoring": "0-5"
    }
}

def build_prompt(rubric, essay_text):
    rubric_info = RUBRIC_GUIDES[rubric]
    return f"""
أنت مقيم لغوي مختص في تقييم مهارة [{rubric_info['arabic']}] في المقالات المكتوبة باللغة العربية.

ستقوم بتقييم المقال بناءً على هذه المهارة فقط.

يرجى اتباع الخطوات التالية:
1. اقرأ المقال جيداً.
2. اتبع دليل التقييم التالي:
{rubric_info['guide']}
3. حدد درجة من {rubric_info['scoring']}.
4. قدم مبررات واضحة لقرارك.

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

def run_model_and_parse_response(model_name, rubric, essay_text):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        device_map="auto"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        trust_remote_code=True,
        device_map="auto"
    )

    model.eval()

    prompt = build_prompt(rubric, essay_text)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=512
        )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
    ]

    decoded_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    output = decoded_output[0]

    print(f"Raw Output:\n {output}")

    try:
        json_data = json.loads(output.split("{", 1)[1].rsplit("}", 1)[0].join(["{", "}"]))
    except Exception as e:
        json_data = {"score": 0, "justification": f"Parsing error: {str(e)}"}
        
    return json_data
