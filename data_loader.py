import json
from langchain.schema import Document


def load_faq_dataset():

    with open(
        "data/faq.json",
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def create_documents_from_faq():

    faq_data = load_faq_dataset()

    documents = []

    synonym_map = {

        "آدرس": [
            "نشانی",
            "مکان",
            "لوکیشن",
            "کجاست"
        ],

        "ایمیل": [
            "پست الکترونیکی",
            "ایمیل رسمی"
        ],

        "شماره تماس": [
            "تلفن",
            "راه ارتباطی"
        ],

        "خوابگاه": [
            "سکونت",
            "اسکان"
        ],

        "سامانه": [
            "سیستم",
            "پرتال"
        ]
    }

    for item in faq_data:

        question = item["question"]

        variants = []

        for key, values in synonym_map.items():

            if key in question:

                variants.extend(values)

        content = (
            f"سوال: {question}\n"
            f"عبارات مشابه: {' ، '.join(variants)}\n"
            f"پاسخ: {item['answer']}"
        )

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": "FAQ Dataset"
                }
            )
        )

    return documents