import json
from pathlib import Path
from typing import Dict, Tuple

def load_data(file_path: str) -> Dict:
    
    with open(file_path, 'r') as f:
        return json.load(f)

def compute_metrics(data: Dict, variety: str, subset: str) -> Tuple[int, float, float]:

    sentiment_data = data.get('Sentiment', {}).get(subset, {}).get(variety, [])
    count = len(sentiment_data)

    if count > 0:
        sentiment_positive = sum(1 for item in sentiment_data if item['label'] == 1)
        sentiment_pct = (sentiment_positive / count) * 100
    else:
        sentiment_pct = 0.0

    sarcasm_data = data.get('Sarcasm', {}).get(subset, {}).get(variety, [])

    if len(sarcasm_data) > 0:
        sarcasm_positive = sum(1 for item in sarcasm_data if item['label'] == 1)
        sarcasm_pct = (sarcasm_positive / len(sarcasm_data)) * 100
    else:
        sarcasm_pct = 0.0

    return count, sentiment_pct, sarcasm_pct

def main():
    data_dir = Path("data/instruction/besstie")

    train_file = data_dir / "train.json"
    test_file = data_dir / "test.json"

    if not train_file.exists() or not test_file.exists():
        print(f"Error: Data files not found in {data_dir}")
        print(f"Expected files: train.json, test.json")
        return

    print("Loading data...")
    train_data = load_data(train_file)
    test_data = load_data(test_file)

    varieties = ['en-AU', 'en-IN', 'en-UK']
    subsets = ['Google', 'Reddit']

    for variety in varieties:
        print(f"\n{variety}:")
        for subset in subsets:
            train_count, train_sent_pct, train_sarc_pct = compute_metrics(train_data, variety, subset)
            test_count, test_sent_pct, test_sarc_pct = compute_metrics(test_data, variety, subset)

            print(f"  {subset}: Train={train_count}, Test={test_count}, "
                  f"Sentiment_Pos={train_sent_pct:.0f}%/{test_sent_pct:.0f}%, "
                  f"Sarcasm_Pos={train_sarc_pct:.0f}%/{test_sarc_pct:.0f}%")
            
    total_train = sum(compute_metrics(train_data, v, s)[0] for v in varieties for s in subsets)
    total_test = sum(compute_metrics(test_data, v, s)[0] for v in varieties for s in subsets)

    print(f"\nTotal Training Samples: {total_train}")
    print(f"Total Test Samples: {total_test}")
    print(f"Total Samples: {total_train + total_test}")

    print("\nBy Variety:")
    for variety in varieties:
        variety_train = sum(compute_metrics(train_data, variety, s)[0] for s in subsets)
        variety_test = sum(compute_metrics(test_data, variety, s)[0] for s in subsets)
        print(f"  {variety}: Train={variety_train}, Test={variety_test}, Total={variety_train + variety_test}")

    print("\nBy Subset:")
    for subset in subsets:
        subset_train = sum(compute_metrics(train_data, v, subset)[0] for v in varieties)
        subset_test = sum(compute_metrics(test_data, v, subset)[0] for v in varieties)
        print(f"  {subset}: Train={subset_train}, Test={subset_test}, Total={subset_train + subset_test}")

    print()

if __name__ == "__main__":
    main()
