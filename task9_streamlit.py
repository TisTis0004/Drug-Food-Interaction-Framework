import os
import streamlit as st
import pandas as pd

# We assume task8_multi_agent_explainer.py is in the same folder
from task8_multi_agent_explainer import (
    explain_interaction_in_arabic,
    DATA_DIR,
    CSV_PATH,
)


@st.cache_data
def load_metadata():
    df = pd.read_csv(CSV_PATH)
    # Unique drug list for dropdown
    drugs = sorted(df["drug_name"].dropna().astype(str).unique())
    return df, drugs


def severity_color(sev: str):
    sev = (sev or "").lower()
    if sev == "severe":
        return "🔴 شديدة"
    elif sev == "moderate":
        return "🟠 متوسطة"
    elif sev == "minor":
        return "🟡 بسيطة"
    elif sev == "info":
        return "🔵 معلوماتية فقط"
    return "⚪ غير معروفة"


def main():
    st.set_page_config(
        page_title="Drug–Food Interaction Explainer (Arabic)", layout="wide"
    )

    st.title("💊🧄 Drug–Food Interaction Explainer")
    st.markdown(
        """
نظام تفاعلي لشرح **تداخلات الأدوية مع الطعام/الأعشاب** باللغة العربية.

> **ملاحظة مهمة:** هذه الأداة لأغراض تعليمية ولا تُغني عن استشارة الطبيب أو الصيدلي.
"""
    )

    df, drugs = load_metadata()

    # Sidebar
    st.sidebar.header("حول النظام")
    st.sidebar.markdown(
        """
- يعتمد على قاعدة بيانات لتداخلات الدواء مع الطعام/الأعشاب.
- يستخدم نماذج تعلّم عميق لتقدير درجة الخطورة.
- يشرح التداخل باللغة العربية بطريقة مبسّطة.
"""
    )

    st.sidebar.subheader("أمثلة جاهزة")
    example = st.sidebar.selectbox(
        "اختر مثالاً:",
        [
            "اختر مثالاً...",
            "Atorvastatin + grapefruit juice",
            "Warfarin + leafy green vegetables",
            "Metformin + food",
            "Fluoxetine + aged cheese",
        ],
    )

    # Default values
    default_drug = None
    default_food = ""
    default_text = ""

    if example == "Atorvastatin + grapefruit juice":
        default_drug = "Atorvastatin"
        default_food = "grapefruit juice"
        default_text = "I am taking atorvastatin. Is it safe to drink grapefruit juice?"
    elif example == "Warfarin + leafy green vegetables":
        default_drug = "Warfarin"
        default_food = "leafy green vegetables"
        default_text = "I am on warfarin and I eat a lot of leafy green vegetables."
    elif example == "Metformin + food":
        default_drug = "Metformin"
        default_food = "food"
        default_text = "My doctor told me to take metformin with food. Why is that?"
    elif example == "Fluoxetine + aged cheese":
        default_drug = "Fluoxetine"
        default_food = "aged cheese"
        default_text = "I'm taking Fluoxetine with aged cheese, is that ok?"

    # Main input area
    st.subheader("أدخل الدواء والطعام/العشبة")

    col1, col2 = st.columns(2)

    with col1:
        drug_name = st.selectbox(
            "اسم الدواء (من القائمة):",
            options=["اختر دواءً..."] + drugs,
            index=(drugs.index(default_drug) + 1 if default_drug in drugs else 0),
        )

    with col2:
        food = st.text_input(
            "الطعام/العشبة (يمكن الكتابة بحرّية):",
            value=default_food,
            help="مثال: grapefruit juice, leafy green vegetables, garlic, alcohol...",
        )

    interaction_text = st.text_area(
        "وصف الاستفسار (اختياري، يمكنك الكتابة بالعربية أو الإنجليزية):",
        value=default_text,
        height=120,
    )

    analyze_button = st.button("تحليل التداخل", type="primary")

    if analyze_button:
        # Basic validation
        if drug_name == "اختر دواءً...":
            st.error("يُرجى اختيار اسم الدواء من القائمة.")
            return
        if not food.strip():
            st.error("يُرجى إدخال اسم الطعام أو العشبة.")
            return

        with st.spinner("جاري تحليل التداخل..."):
            result = explain_interaction_in_arabic(
                drug_name=drug_name,
                food=food,
                interaction_text=interaction_text,
                k=10,
            )

        # If not relevant
        if not result.get("relevant", True):
            st.warning(
                "❗ تم اعتبار هذا الاستفسار خارج نطاق التداخلات الدوائية مع الطعام/الأعشاب."
            )
            st.markdown(result["arabic_explanation"])
            return

        # Relevant case
        severity_info = result["severity_info"]
        neighbors = result["neighbors"]
        arabic_explanation = result["arabic_explanation"]

        # Top section: severity
        st.subheader("نتيجة التقييم")
        sev_label = severity_info["final_severity"]
        confidence = severity_info["confidence_level"]
        st.markdown(f"**درجة الخطورة:** {severity_color(sev_label)}")
        st.markdown(f"**درجة الثقة في التقييم:** `{confidence}`")

        st.markdown("---")
        st.subheader("الشرح باللغة العربية")
        st.markdown(arabic_explanation)

        # Details (expander)
        with st.expander("تفاصيل النماذج والأدلة (للمتخصصين)"):
            st.markdown("**تجميع الشدة (Severity Aggregation)**")
            st.json(
                {
                    "final_severity": severity_info["final_severity"],
                    "neighbor_pred": severity_info["neighbor_pred"],
                    "neighbor_confidence": severity_info["neighbor_confidence"],
                    "neighbor_distribution": severity_info["neighbor_distribution"],
                    "classifier_pred": severity_info["classifier_pred"],
                    "classifier_probs": severity_info["classifier_probs"],
                    "confidence_level": severity_info["confidence_level"],
                }
            )

            st.markdown("---")
            st.markdown("**أقرب الحالات المشابهة في قاعدة البيانات**")

            if neighbors:
                df_neighbors = pd.DataFrame(neighbors)
                # Show a subset of columns
                cols_to_show = [
                    "drug_name",
                    "food",
                    "severity",
                    "interaction_text",
                    "distance",
                    "similarity",
                ]
                cols_to_show = [c for c in cols_to_show if c in df_neighbors.columns]
                st.dataframe(df_neighbors[cols_to_show])
            else:
                st.write("لا توجد حالات مشابهة متاحة.")

    st.markdown("---")
    st.caption(
        "⚠️ هذه الأداة لأغراض تعليمية ولا تُعتبَر بديلاً عن استشارة الطبيب أو الصيدلي."
    )


if __name__ == "__main__":
    main()
