import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(11, 7.5))
ax.set_xlim(0, 11)
ax.set_ylim(0, 9.5)
ax.axis("off")

actors = [
    ("User", 0.8),
    ("main.py\n(CLI)", 3.0),
    ("prediction_\nmodule.py", 5.3),
    ("config.py", 7.3),
    ("rf_model.joblib\n(saved model)", 9.6),
]

top_y = 9.0
bottom_y = 0.6

# lifelines
for name, x in actors:
    ax.add_patch(mpatches.FancyBboxPatch((x-0.9, top_y-0.35), 1.8, 0.6,
                 boxstyle="round,pad=0.02", fc="#EAF2FB", ec="#2C5F8A", lw=1.2))
    ax.text(x, top_y-0.05, name, ha="center", va="center", fontsize=9, fontweight="bold")
    ax.plot([x, x], [top_y-0.35, bottom_y], color="#888888", lw=1, ls="--")

def arrow(y, x_from, x_to, label, dashed=False, color="#333333"):
    style = "->"
    ls = "dashed" if dashed else "solid"
    a = FancyArrowPatch((x_from, y), (x_to, y), arrowstyle=style, mutation_scale=14,
                         color=color, lw=1.3, linestyle=ls)
    ax.add_patch(a)
    mid = (x_from + x_to) / 2
    ax.text(mid, y+0.12, label, ha="center", va="bottom", fontsize=8)

user_x = actors[0][1]
main_x = actors[1][1]
pred_x = actors[2][1]
cfg_x = actors[3][1]
model_x = actors[4][1]

y = 8.1
arrow(y, user_x, main_x, "python main.py predict --input patient.csv")
y -= 0.75
arrow(y, main_x, pred_x, "predict_patients(input_path)")
y -= 0.75
arrow(y, pred_x, cfg_x, "read ALL_GENES, POSITIVE_CLASS")
y -= 0.6
arrow(y, cfg_x, pred_x, "gene panel, class labels", dashed=True)
y -= 0.75
arrow(y, pred_x, pred_x+0.01, "", dashed=True)
ax.text(pred_x+0.15, y+0.05, "validate_input(df)\n[checks columns, nulls]", fontsize=8, va="center")
y -= 0.9
arrow(y, pred_x, model_x, "load_model()")
y -= 0.6
arrow(y, model_x, pred_x, "trained RandomForestClassifier", dashed=True)
y -= 0.75
arrow(y, pred_x, model_x, "model.predict(X), model.predict_proba(X)")
y -= 0.6
arrow(y, model_x, pred_x, "predicted class + probabilities", dashed=True)
y -= 0.75
ax.text(pred_x+0.15, y+0.15, "rank top contributing genes\n(importance x expression)", fontsize=8, va="center")
y -= 0.55
arrow(y, pred_x, main_x, "prediction results (list[dict])", dashed=True)
y -= 0.75
arrow(y, main_x, user_x, "printed JSON: subtype, probabilities,\ntop genes", dashed=True)

ax.text(5.3, 9.4, "Sequence Diagram -- Predict Patient Subtype", ha="center", fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig("/home/claude/oscc-ms-real/docs/diagrams/sequence.png", dpi=150, bbox_inches="tight")
print("saved")
