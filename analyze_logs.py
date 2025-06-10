import json, pathlib, statistics, csv
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

TASKS_GOLD = [
    {"A1": "Partly Helpful", "A2": "Helpful", "A3": "Unhelpful"},
    {"B1": "Helpful", "B2": "Partly Helpful", "B3": "Unhelpful"},
    {"C1": "Helpful", "C2": "Unhelpful", "C3": "Partly Helpful"},
    {"D1": "Partly Helpful", "D2": "Helpful", "D3": "Unhelpful"},
    {"E1": "Unhelpful", "E2": "Partly Helpful", "E3": "Helpful"},
]


def process_log(path: pathlib.Path) -> dict:
    with open(path) as f:
        events = json.load(f)

    t_start = next(e["t"] for e in events if e["ev"] == "timer_start")
    t_end = next(e["t"] for e in reversed(events) if e["ev"] == "task_complete")
    total_time = (t_end - t_start) / 1000 

    loads = {e["task"]: e["t"] for e in events if e["ev"] == "task_load"}
    ordered = sorted(loads.items())
    micro = [(ordered[i + 1][1] if i + 1 < len(ordered) else t_end) - t0
             for i, (_, t0) in enumerate(ordered)]
    micro = [d / 1000 for d in micro]

    sel_count = {}
    for e in events:
        if e["ev"] == "label_select":
            sel_count[(e["task"], e["answerId"])] = sel_count.get((e["task"], e["answerId"]), 0) + 1

    mismatches, changes, confirms = 0, 0, []
    confirms = [e for e in events if e["ev"] == "label_confirm"]
    for e in confirms:
        if e["label"] != TASKS_GOLD[e["task"]][e["answerId"]]:
            mismatches += 1
        if sel_count.get((e["task"], e["answerId"]), 1) > 1:
            changes += 1

    guideline_uses = sum(1 for e in events if e["ev"] == "sidebar_open")
    blur_events = sum(1 for e in events if e["ev"] == "blur")
    distraction_ratio = blur_events / len(events)

    return {
        "participant": path.stem[len(path.stem) - 2:],
        "total_time_s": round(total_time,2),
        "mean_micro_task_s": round(statistics.mean(micro),2),
        "mislabel_rate": round(mismatches / len(confirms),2),
        "decision_change_rate": round(changes / len(confirms),2),
        "guideline_uses": guideline_uses,
        "distraction_ratio": round(distraction_ratio,2),
    }, micro


metrics, all_micro = [], []
for fp in pathlib.Path(".").glob("log_*.json"):
    row, micro = process_log(fp)
    metrics.append(row)
    all_micro.extend(micro)

df = pd.DataFrame(metrics).set_index("participant")
df.to_csv("results_summary.csv")

print(df.mean())
print(df.std())

print(df.round(3), "\nSaved: results_summary.csv")

plt.figure()
df["total_time_s"].plot(kind="bar")
plt.ylabel("seconds")
plt.title("Completion time")
plt.tight_layout()
plt.savefig("time_per_participant.png")

plt.figure()
df["mislabel_rate"].plot(kind="bar")
plt.ylabel("rate")
plt.title("Mislabeling rate")
plt.tight_layout()
plt.savefig("mislabel_rate.png")

plt.figure()
plt.hist(all_micro, bins=10)
plt.xlabel("seconds")
plt.title("Micro-task durations")
plt.tight_layout()
plt.savefig("micro_durations.png")

plt.figure()
plt.scatter(df["decision_change_rate"], df["mislabel_rate"], color='blue')

for i, participant in enumerate(df.index):
    plt.annotate(participant,
                 (df["decision_change_rate"].iloc[i], df["mislabel_rate"].iloc[i]),
                 textcoords="offset points", xytext=(5,5), ha='right', fontsize=8)

plt.xlabel("Decision Change Rate")
plt.ylabel("Mislabel Rate")
plt.title("Decision change vs Mislabeling")
plt.grid(True)
plt.tight_layout()
plt.savefig("decision_vs_mislabel.png")

plt.figure()
plt.scatter(df["guideline_uses"], df["mislabel_rate"])

for i, participant in enumerate(df.index):
    plt.annotate(participant,
                 (df["guideline_uses"].iloc[i], df["mislabel_rate"].iloc[i]),
                 textcoords="offset points", xytext=(5,5), ha='right', fontsize=8)

plt.xlabel("Guidelines Uses")
plt.ylabel("Mislabel Rate")
plt.title("Impact of using Guidelines on Label Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig("guidelines_mislabel.png")

plt.figure()
df["mean_micro_task_s"].plot(kind="bar")
plt.ylabel("Mean Micro-task Duration (s)")
plt.title("Mean Micro-task Time per Participant")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("micro_task_time_per_participant.png")

print("Plots saved: *.png")
