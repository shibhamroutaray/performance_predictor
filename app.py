# quick student grade prediction thing (streamlit)
# not cleaned up properly but works fine

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap

# Load model (hope file exists lol)
try:
    with open("rf_model.pkl", "rb") as f:
        model = pickle.load(f)
except:
    st.error("Model file missing (rf_model.pkl)")
    st.stop()


# ------------------------------- feature prep ---------------------

def make_feats(studytime, absences, g1, g2, medu, fedu,
               wknd_alc, wkday_alc, soc, free_time):

    # model wants dataframe, not dict
    d = pd.DataFrame({
        "studytime": [studytime],
        "absences": [absences],
        "Medu": [medu],
        "Fedu": [fedu],
        "G1": [g1],
        "G2": [g2],
        "weekend_alcohol": [wknd_alc],
        "weekday_alcohol": [wkday_alc],
        "social_activity": [soc],
        "free_time_score": [free_time],
    })

    # some quick engineered stuff
    # (can tune later, don't overthink)
    if studytime == 0:
        eff = (g1 + g2)
    else:
        eff = (g1 + g2) / studytime

    d["study_efficiency"] = eff
    d["attendance_ratio"] = 1/(1+absences)   # meh formula but ok
    d["parent_edu_index"] = (medu + fedu) / 2

    # column order matters (annoying)
    ordered = [
        "studytime", "absences", "Medu", "Fedu", "G1", "G2",
        "study_efficiency", "attendance_ratio", "parent_edu_index",
        "weekend_alcohol", "weekday_alcohol",
        "social_activity", "free_time_score"
    ]

    return d[ordered]


# ---------------------------- UI part -----------------------------

st.title("Student Grade Predictor (rough version)")
st.write("Put values below & it’ll estimate final G3. Nothing fancy.")

st.subheader("Student Inputs")

study = st.slider("Study Time", 1, 4, 2)
absn = st.slider("Absences", 0, 30, 3)
g1 = st.slider("G1", 0, 20, 10)
g2 = st.slider("G2", 0, 20, 12)
medu = st.slider("Mother Education (0-4)", 0, 4, 2)
fedu = st.slider("Father Education (0-4)", 0, 4, 2)

wknd = st.slider("Weekend Alcohol (1-5)", 1, 5, 1)
wkday = st.slider("Weekday Alcohol (1-5)", 1, 5, 1)
soc = st.slider("Social Activity (1-5)", 1, 5, 3)
free_t = st.slider("Free Time Quality (1-5)", 1, 5, 3)


# --------------------------- prediction ----------------------------

if st.button("Predict"):
    sample = make_feats(study, absn, g1, g2, medu, fedu,
                        wknd, wkday, soc, free_t)

    try:
        pred = model.predict(sample)[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    st.success(f"Predicted G3: {pred:.2f}")

    st.write("### SHAP Explanation (why this prediction?)")

    try:
        expl = shap.TreeExplainer(model)
        sv = expl.shap_values(sample)

        # shap plotting
        fig, ax = plt.subplots()
        shap.waterfall_plot(
            shap.Explanation(
                values=sv[0],
                base_values=expl.expected_value,
                data=sample.iloc[0],
                feature_names=sample.columns
            ),
            max_display=13
        )
        st.pyplot(fig)

    except Exception as e:
        st.warning("SHAP failed to run, maybe missing dependencies?")
        st.text(str(e))
