
import re
import time
import pandas as pd
from pathlib import Path

import ocr_orchesterator as router

DATA_DIR = Path("datasets")
IMAGES_DIR = DATA_DIR / "eval_images"
SAMPLES_PER_TYPE = 100

IMAGES_DIR.mkdir(parents=True, exist_ok=True)



def normalize(text):
    text = text.strip()

    # Convert Arabic-Indic digits to Western digits for evaluation.
    text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))

    # Normalize whitespace around punctuation.
    text = re.sub(r"\s*([/.,:;()-])\s*", r"\1", text)
    text = re.sub(r"\s+", " ", text)

    return text



def edit_distance(a, b):
    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, 1):
        current = [i]

        for j, char_b in enumerate(b, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (char_a != char_b),
            ))

        previous = current

    return previous[-1]


def cer(prediction, truth):
    prediction = normalize(prediction)
    truth = normalize(truth)

    if not truth:
        return 0.0 if not prediction else 1.0

    return edit_distance(prediction, truth) / len(truth)


datasets = {
    "arabic": DATA_DIR / "test_ar.parquet",
    "english": DATA_DIR / "test_en.parquet",
    "numbers": DATA_DIR / "test_numbers.parquet",
}

records = []

for category, file_path in datasets.items():
    df = pd.read_parquet(file_path)
    sample = df.sample(
        n=min(SAMPLES_PER_TYPE, len(df)),
        random_state=42,
    )

    print(f"\nTesting {category}: {len(sample)} images")

    for index, row in sample.iterrows():
        image_path = IMAGES_DIR / (
            f"{category}_{row['image_name']}.png"
        )
        image_path.write_bytes(bytes(row["image_bytes"]))

        truth = str(row["full_text"]).strip()

        # Arabic model only
        start = time.perf_counter()
        arabic = router.extract_prediction(
            router.arabic_ocr, str(image_path)
        )
        arabic_time = time.perf_counter() - start

        # English model only
        start = time.perf_counter()
        english = router.extract_prediction(
            router.get_english_ocr(), str(image_path)
        )
        english_time = time.perf_counter() - start

        # Current router
        start = time.perf_counter()
        selected, _, _ = router.process_crop(str(image_path))
        router_time = time.perf_counter() - start

        records.append({
            "category": category,
            "image": image_path.name,
            "ground_truth": truth,
            "arabic_text": arabic["text"],
            "arabic_cer": cer(arabic["text"], truth),
            "arabic_seconds": arabic_time,
            "english_text": english["text"],
            "english_cer": cer(english["text"], truth),
            "english_seconds": english_time,
            "router_text": selected["text"],
            "router_cer": cer(selected["text"], truth),
            "router_model": selected["model"],
            "router_review": selected["needs_review"],
            "router_seconds": router_time,
        })

        print(
            f"{image_path.name}: "
            f"Arabic CER={cer(arabic['text'], truth):.1%}, "
            f"English CER={cer(english['text'], truth):.1%}, "
            f"Router CER={cer(selected['text'], truth):.1%}"
        )

results = pd.DataFrame(records)
output_path = DATA_DIR / "evaluation_results.csv"
results.to_csv(output_path, index=False, encoding="utf-8-sig")

print("\n=== Average results by category ===")

summary = results.groupby("category").agg(
    Arabic_CER=("arabic_cer", "mean"),
    English_CER=("english_cer", "mean"),
    Router_CER=("router_cer", "mean"),
    Arabic_seconds=("arabic_seconds", "mean"),
    English_seconds=("english_seconds", "mean"),
    Router_seconds=("router_seconds", "mean"),
)

print((summary * 1).round(3).to_string())
print(f"\nDetailed results saved to: {output_path}")
