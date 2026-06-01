"""Stratified 30-case subset of qa.large.yaml: 6 cases from each of 5 domains."""
import yaml
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)
src = yaml.safe_load(open("docs/blog/article4/qa.large.yaml"))
by_domain = defaultdict(list)
for c in src["cases"]:
    by_domain[c["domain"]].append(c)

subset = []
for domain, cases in sorted(by_domain.items()):
    n_pick = min(6, len(cases))
    picks = random.sample(cases, n_pick)
    subset.extend(picks)
print(f"subset by domain: { {d: len(by_domain[d]) for d in by_domain} }")
print(f"picked total: {len(subset)}")
ds = {"name": "qa-small-stratified-v1", "cases": subset}
Path("docs/blog/article4/qa.small.yaml").write_text(
    yaml.safe_dump(ds, allow_unicode=True, sort_keys=False, width=120)
)
print("wrote docs/blog/article4/qa.small.yaml")
