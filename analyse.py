import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# 1. Load Data
df = pd.read_csv('Final-CSV_Cognitive_Science.csv')

# Keep only Male / Female
df = df[df["Gender"].isin(["Male", "Female"])]

# ----- AI USAGE: MAP TO ORDINAL NUMERIC (1–6) -----
ai_map = {
    "Never": 1,
    "Used it Few times": 2,
    "Rarely (Few times a month)": 3,
    "Constant (Few times a week)": 4,
    "Everyday": 5,
    "Multiple Times a Day": 6
}

df["AI_Exposure_clean"] = df["AI_Exposure"].astype(str).str.strip()
df["AI_Exposure_Num"] = df["AI_Exposure_clean"].map(ai_map)

# ----- EDUCATION: MAP TO ORDINAL NUMERIC (1–8) -----
education_map = {
    "Middle school": 1,
    "High school": 2,
    "Started university (Bachelors)": 3,
    "Completed university (Bachelors)": 4,
    "Started masters": 5,
    "Completed masters": 6,
    "P.h.d.": 7,
    "Technical School": 8
}

df["Education_clean"] = df["Education"].astype(str).str.strip()
df["Education_Num"] = df["Education_clean"].map(education_map)

# ----- CLAIM TRUTHNESS MAP -----
truth_map = {
    "Q1": 0, "Q2": 0, "Q3": 1, "Q4": 0, "Q5": 0,
    "Q6": 0, "Q7": 0, "Q8": 1, "Q9": 1, "Q10": 0,
    "Q11": 1, "Q12": 1, "Q13": 0, "Q14": 0, "Q15": 1,
    "Q16": 1, "Q17": 0, "Q18": 0, "Q19": 0, "Q20": 0
}

# 2. LONG FORMAT EXPANSION
long_data = []
for index, row in df.iterrows():
    for i in range(1, 21):
        long_data.append({
            'Participant_ID': row['Participant_ID'],
            'Gender': row['Gender'],
            'AI_Exposure_Num': row["AI_Exposure_Num"],
            'Education_Num': row["Education_Num"],
            'Claim_ID': f'Q{i}',
            'Pre_Rating_Text': row[f'Q{i}-pre'],
            'Source': row[f'Q{i}-Source'],
            'Advice_Label_Text': row[f'Q{i}-Label'],
            'Post_Rating_Text': row[f'Q{i}-post']
        })

df_long = pd.DataFrame(long_data)

# 3. MAP TEXT RATINGS TO NUMERIC
rating_map = {
    "true": 6, "likely true": 5, "possibly true": 4,
    "possibly false": 3, "likely false": 2, "false": 1
}

def map_rating(value):
    if pd.isna(value):
        return np.nan
    return rating_map.get(str(value).strip().lower(), np.nan)

df_long['Pre_Num'] = df_long['Pre_Rating_Text'].apply(map_rating)
df_long['Post_Num'] = df_long['Post_Rating_Text'].apply(map_rating)
df_long['Advice_Num'] = df_long['Advice_Label_Text'].apply(map_rating)

# 4. COMPUTE DISTANCES
df_long['Abs_Diff_to_Advice'] = abs(df_long['Post_Num'] - df_long['Advice_Num'])
df_long['Initial_Distance'] = abs(df_long['Pre_Num'] - df_long['Advice_Num'])

# Clean missing
df_long_clean = df_long.dropna(subset=[
    'Abs_Diff_to_Advice', 'Initial_Distance', 'Source',
    'Gender', 'AI_Exposure_Num', 'Education_Num'
])

# Convert to categorical types
df_long_clean["Gender"] = df_long_clean["Gender"].astype("category")
df_long_clean["Source"] = df_long_clean["Source"].astype("category")

# Add Truth label
df_long_clean["Truth"] = df_long_clean["Claim_ID"].map(truth_map).astype(int)

# 5. MAIN MIXED MODEL WITH:
#    - participant random intercept
#    - participant random slope for Initial_Distance
#    - claim random intercept
#    - fixed effects: Truth, Truth×Source
formula = """
Abs_Diff_to_Advice ~ 
    Initial_Distance * Source + 
    Education_Num +
    AI_Exposure_Num +
    Gender
"""


print("\n=== MIXED MODEL WITH CLAIM & PARTICIPANT RANDOM COMPONENTS ===")
model = smf.mixedlm(
    formula,
    data=df_long_clean,
    groups=df_long_clean["Participant_ID"],
    re_formula="~Initial_Distance",
    vc_formula={"Claim": "0 + C(Claim_ID)"}
)

result = model.fit(method="lbfgs")
print(result.summary())


# Compute Advice Weight (AW)
def compute_aw(pre, post, adv):
    if pd.isna(pre) or pd.isna(post) or pd.isna(adv):
        return np.nan
    denom = adv - pre
    if denom == 0:
        return 0
    return (post - pre) / denom

df_long_clean["AW"] = df_long_clean.apply(
    lambda row: compute_aw(row["Pre_Num"], row["Post_Num"], row["Advice_Num"]),
    axis=1
)

df_aw = df_long_clean.dropna(subset=["AW"])


formula_aw = """
AW ~ 
    Initial_Distance * Source+
    Education_Num +
    AI_Exposure_Num
"""

print("\n=== MIXED MODEL PREDICTING ADVICE WEIGHT (AW) ===")

model_aw = smf.mixedlm(
    formula_aw,
    data=df_aw,
    groups=df_aw["Participant_ID"],
    re_formula="~Initial_Distance",
    vc_formula={"Claim": "0 + C(Claim_ID)"}
)

result_aw = model_aw.fit(method="lbfgs")
print(result_aw.summary())









import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# LOAD AND LONG-FORM TRANSFORM
# -----------------------------
df = pd.read_csv("Final-CSV_Cognitive_Science.csv")

rating_map = {
    "true": 6, "likely true": 5, "possibly true": 4,
    "possibly false": 3, "likely false": 2, "false": 1
}

def to_num(x):
    if pd.isna(x):
        return None
    return rating_map.get(str(x).strip().lower(), None)

records = []
for _, row in df.iterrows():
    pid = row["Participant_ID"]
    gender = row["Gender"]
    for i in range(1, 21):
        pre = to_num(row[f"Q{i}-pre"])
        post = to_num(row[f"Q{i}-post"])
        adv = to_num(row[f"Q{i}-Label"])
        src = row[f"Q{i}-Source"]
        if pre is None or post is None or adv is None:
            continue
        denom = adv - pre
        aw = np.nan
        if denom != 0:
            aw = (post - pre) / denom

        records.append({
            "Participant": pid,
            "Gender": gender,
            "Source": src,
            "Pre": pre,
            "Post": post,
            "Advice": adv,
            "Initial_Distance": abs(pre - adv),
            "Final_Distance": abs(post - adv),
            "Adjustment": post - pre,
            "Advice_Weight": aw
        })

df_long = pd.DataFrame(records)

# ==========================
# PLOT 1 – MAIN EFFECT: Initial Distance × Source
# ==========================

# Bin initial distance
df_long["ID_bin"] = pd.cut(
    df_long["Initial_Distance"],
    bins=[0,1,2,3,4,5],
    labels=["0–1","1–2","2–3","3–4","4–5"],
    include_lowest=True
)

agg = df_long.groupby(["ID_bin","Source"]).agg(
    mean_final=("Final_Distance","mean"),
    se_final=("Final_Distance", lambda x: x.std(ddof=1) / np.sqrt(len(x)))
).reset_index()

pivot_mean = agg.pivot(index="ID_bin", columns="Source", values="mean_final")
pivot_se   = agg.pivot(index="ID_bin", columns="Source", values="se_final")

plt.figure(figsize=(7,5))
x = np.arange(len(pivot_mean.index))

for j, src in enumerate(pivot_mean.columns):
    y = pivot_mean[src].values
    yerr = pivot_se[src].values
    plt.errorbar(x + 0.03*j, y, yerr=yerr, fmt='-o', label=src)

plt.xticks(x, pivot_mean.index)
plt.xlabel("Initial Distance Bin (|Pre - Advice|)")
plt.ylabel("Mean Final Distance to Advice")
plt.title("Effect of Initial Disagreement by Source (AI vs Human)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Interpretation:
# - Lines diverging as you move right = AI more resilient than Human for large disagreement.

# ==========================================
# PLOT 2 – Source × Gender (Male vs Female only)
# ==========================================

df_gender = df_long[df_long["Gender"].isin(["Male", "Female"])]

group2 = df_gender.groupby(["Gender","Source"])["Final_Distance"].mean().unstack()

plt.figure(figsize=(6,4))
group2.plot(kind="bar", color=["tab:blue","tab:orange"])

plt.ylabel("Mean Final Distance to Advice")
plt.title("Source × Gender: Final Distance (Male vs Female Only)")
plt.xticks(rotation=0)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# Interpretation:
# - Higher bar = stays further from advice (updates less).
# - You should see men generally higher.

# ==========================
# PLOT 3 – Distribution of Advice Weight (WoA) by Source
# ==========================

df_aw = df_long.dropna(subset=["Advice_Weight"])

plt.figure(figsize=(7,5))
for src in df_aw["Source"].unique():
    sub = df_aw[df_aw["Source"] == src]
    plt.hist(sub["Advice_Weight"], bins=20, alpha=0.5, label=src)

plt.xlabel("Advice Weight ( (Post - Pre) / (Advice - Pre) )")
plt.ylabel("Count")
plt.title("Distribution of Advice Weight by Source")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


rating_map = {
    "true": 6, "likely true": 5, "possibly true": 4,
    "possibly false": 3, "likely false": 2, "false": 1
}

def to_num(x):
    if pd.isna(x):
        return None
    return rating_map.get(str(x).strip().lower(), None)

rows=[]
for _, row in df.iterrows():
    pid=row["Participant_ID"]
    for i in range(1,21):
        pre=to_num(row[f"Q{i}-pre"])
        post=to_num(row[f"Q{i}-post"])
        adv=to_num(row[f"Q{i}-Label"])
        src=row[f"Q{i}-Source"]
        if pre is None or post is None or adv is None:
            continue
        denom = adv - pre
        if denom != 0:
            aw = (post - pre) / denom
        else:
            aw = None
        rows.append({"AW": aw, "Source": src})

df_aw = pd.DataFrame(rows).dropna(subset=["AW"])

# Filter to AW ≠ 0
df_aw_nz = df_aw[df_aw["AW"] != 0]

# Plot 3.1
plt.figure(figsize=(7,5))
for src in ["AI","Human"]:
    subset = df_aw_nz[df_aw_nz["Source"] == src]
    plt.hist(subset["AW"], bins=25, alpha=0.5, label=src)

plt.xlabel("Advice Weight (non-zero only)")
plt.ylabel("Count")
plt.title("Distribution of Advice Weight (AW ≠ 0) by Source")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

df = pd.read_csv("Final-CSV_with_ParticipantMetrics.csv")
# Rating map
rating_map = {
    "true": 6, "likely true": 5, "possibly true": 4,
    "possibly false": 3, "likely false": 2, "false": 1
}

def to_num(x):
    if pd.isna(x): 
        return None
    return rating_map.get(str(x).strip().lower(), None)

# Build long-format AW data
records=[]
for _, row in df.iterrows():
    pid=row["Participant_ID"]
    stability=row["Stability"]
    for i in range(1,21):
        pre = to_num(row[f"Q{i}-pre"])
        post = to_num(row[f"Q{i}-post"])
        adv = to_num(row[f"Q{i}-Label"])
        src = row[f"Q{i}-Source"]
        if pre is None or post is None or adv is None:
            continue
        denom = adv - pre
        if denom != 0:
            aw = (post - pre) / denom
        else:
            aw = None
        
        records.append({
            "Participant": pid,
            "Source": src,
            "AW": aw,
            "Stability": stability
        })

df_aw = pd.DataFrame(records).dropna(subset=["AW"])

# Create stability groups based on quantiles
low_thr  = df_aw["Stability"].quantile(0.33)
high_thr = df_aw["Stability"].quantile(0.66)

def group_stability(x):
    if x <= low_thr:
        return "Low Stability\n(Highly Adaptive)"
    elif x <= high_thr:
        return "Medium Stability"
    else:
        return "High Stability\n(Highly Resistant)"

df_aw["Stability_Group"] = df_aw["Stability"].apply(group_stability)

# FIXED: enforce correct order explicitly
groups = [
    "Low Stability\n(Highly Adaptive)",
    "Medium Stability",
    "High Stability\n(Highly Resistant)"
]

# ====================================
# Plot AW distribution separately per group
# ====================================

fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)

for ax, group in zip(axes, groups):
    sub = df_aw[df_aw["Stability_Group"] == group]

    ax.hist(sub[sub["Source"]=="AI"]["AW"], bins=25, alpha=0.5,
            color="orange", label="AI Advice")

    ax.hist(sub[sub["Source"]=="Human"]["AW"], bins=25, alpha=0.5,
            color="blue", label="Human Advice")

    ax.set_title(f"Advice Weight Distribution – {group}")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)
    ax.legend()

plt.xlabel("Advice Weight")
plt.tight_layout()
plt.show()




df = pd.read_csv("Final-CSV_with_ParticipantMetrics.csv")

# Rating map
rating_map = {
    "true": 6, "likely true": 5, "possibly true": 4,
    "possibly false": 3, "likely false": 2, "false": 1
}

def to_num(x):
    if pd.isna(x): 
        return None
    return rating_map.get(str(x).strip().lower(), None)

# Build long-format AW data
records=[]
for _, row in df.iterrows():
    pid=row["Participant_ID"]
    stability=row["Stability"]
    for i in range(1,21):
        pre = to_num(row[f"Q{i}-pre"])
        post = to_num(row[f"Q{i}-post"])
        adv = to_num(row[f"Q{i}-Label"])
        src = row[f"Q{i}-Source"]
        if pre is None or post is None or adv is None:
            continue
        denom = adv - pre
        if denom != 0:
            aw = (post - pre) / denom
        else:
            aw = None
        
        records.append({
            "Participant": pid,
            "Source": src,
            "AW": aw,
            "Stability": stability
        })

df_aw = pd.DataFrame(records).dropna(subset=["AW"])
df_aw = df_aw[df_aw["AW"] != 0]

# Create stability groups based on quantiles
low_thr  = df_aw["Stability"].quantile(0.33)
high_thr = df_aw["Stability"].quantile(0.66)

def group_stability(x):
    if x <= low_thr:
        return "Low Stability\n(Adaptive)"
    elif x <= high_thr:
        return "Medium Stability"
    else:
        return "High Stability\n(Resistant)"

df_aw["Stability_Group"] = df_aw["Stability"].apply(group_stability)

groups = ["Low Stability\n(Adaptive)", "Medium Stability", "High Stability\n(Resistant)"]

# Plot: AW distributions separated by Stability Group and Source
fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)

for ax, group in zip(axes, groups):
    sub = df_aw[df_aw["Stability_Group"] == group]

    ax.hist(sub[sub["Source"]=="AI"]["AW"], bins=30, alpha=0.5,
            color="orange", label="AI Advice")

    ax.hist(sub[sub["Source"]=="Human"]["AW"], bins=30, alpha=0.5,
            color="blue", label="Human Advice")

    ax.set_title(f"Advice Weight Distribution – {group}")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)
    ax.legend()

plt.xlabel("Advice Weight (AW ≠ 0)")
plt.tight_layout()
plt.show()