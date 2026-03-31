# check_review.py
from guardrails.provenance import read_log
import os, json

logs = sorted(os.listdir('outputs/provenance'))
run_id = logs[-1].replace('.jsonl', '')
print("Run ID:", run_id)

events = read_log(run_id)

for e in events:
    if e['event'] == 'AWAITING_HUMAN_APPROVAL':
        data = e['data']
        review = data.get('review', {})
        for r in review.get('reviews', []):
            print(f"[{r['support_level']}] {r['original_statement'][:100]}")
            print(f"  -> {r['reviewer_comment']}")
            print()
        print("Overall verdict:", review.get('overall_verdict'))