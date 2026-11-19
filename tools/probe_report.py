"""Print velocity disagreement by adjacent window pair across sampler steps."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def report(path):
    overlaps = defaultdict(lambda: defaultdict(list))
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            overlaps[row["overlap_index"]][row["step"]].append(row)

    for index, by_step in sorted(overlaps.items()):
        print(f"Overlap {index} (windows {index} and {index + 1})")
        print("step    low       mid       high      rel_l2    evaluations")
        previous = stopped = None
        for step, rows in sorted(by_step.items(), key=lambda item: (item[0] is None, item[0])):
            values = []
            for band in ("low", "mid", "high", "rel_l2"):
                samples = [row[band] for row in rows]
                values.append(mean(samples) if None not in samples else None)
            columns = "  ".join(f"{v:8.5f}" if v is not None else "     n/a" for v in values)
            print(f"{str(step):>4} {columns}  {len(rows)}")
            low = values[0]
            if step is not None and stopped is None and previous is not None and low is not None and low >= previous:
                stopped = step
            previous = low
        print("First step where low stopped falling:",
              stopped if stopped is not None else "none observed")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe_jsonl", type=Path)
    report(parser.parse_args().probe_jsonl)
