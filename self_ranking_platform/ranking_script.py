import streamlit as st
import pandas as pd
import random
from pathlib import Path

# Configure page
st.set_page_config(page_title="News Headline Annotator", layout="wide")

# Initialize session state
if 'current_headline' not in st.session_state:
    st.session_state.current_headline = None
if 'annotations' not in st.session_state:
    st.session_state.annotations = []
if 'df' not in st.session_state:
    st.session_state.df = None
if 'annotated_indices' not in st.session_state:
    st.session_state.annotated_indices = set()

# Categories for relevancy rating
CATEGORIES = [
    "Politics & Government",
    "Economy & Finance",
    "Security & Military",
    "Health & Medicine",
    "Science & Climate",
    "Technology"
]


def load_data(file):
    """Load the CSV dataset"""
    df = pd.read_csv(file)
    return df


def get_random_headline(df, annotated_indices):
    """Get a random headline that hasn't been annotated yet"""
    available_indices = set(df.index) - annotated_indices
    if not available_indices:
        return None
    idx = random.choice(list(available_indices))
    return idx, df.loc[idx]


def save_annotations(annotations, output_path):
    """Save annotations to CSV"""
    if annotations:
        df_out = pd.DataFrame(annotations)
        df_out.to_csv(output_path, index=False)
        return True
    return False


# Main UI
st.title("📰 News Headline Annotation Tool")
st.markdown("Rate headlines on sentiment and relevancy across multiple categories")

# File upload
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=['csv'])

if uploaded_file is not None:
    if st.session_state.df is None:
        st.session_state.df = load_data(uploaded_file)
        st.success(f"✅ Loaded {len(st.session_state.df)} headlines")

    # Progress indicator
    total = len(st.session_state.df)
    annotated = len(st.session_state.annotated_indices)
    st.progress(annotated / total if total > 0 else 0)
    st.write(f"Progress: {annotated}/{total} headlines annotated")

    # Load next headline button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("📋 Load Next Headline", use_container_width=True):
            result = get_random_headline(st.session_state.df, st.session_state.annotated_indices)
            if result:
                st.session_state.current_headline = result
            else:
                st.warning("🎉 All headlines have been annotated!")
                st.session_state.current_headline = None

    # Display current headline
    if st.session_state.current_headline:
        idx, row = st.session_state.current_headline

        st.markdown("---")
        st.markdown("### Current Headline")

        # Display headline info
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.write(f"**Date:** {row['date']}")
        with info_col2:
            st.write(f"**Source:** {row['source']}")
        with info_col3:
            st.write(f"**Importance:** {row['importance_level']}")

        st.markdown(f"### 📰 *\"{row['headline']}\"*")
        st.markdown("---")

        # Rating form
        with st.form("annotation_form"):
            st.markdown("#### 📊 Rate this headline")

            # Sentiment rating
            st.markdown("**Reader Sentiment** (-5 = Very Negative, 0 = Neutral, 5 = Very Positive)")
            sentiment = st.slider("Sentiment", -5, 5, 0, key="sentiment")

            st.markdown("---")
            st.markdown("**Relevancy Ratings** (0 = Not Relevant, 5 = Highly Relevant)")

            # Category ratings
            ratings = {}
            for category in CATEGORIES:
                ratings[category] = st.slider(category, 0, 5, 0, key=category)

            # Submit button
            submitted = st.form_submit_button("✅ Submit Rating", use_container_width=True)

            if submitted:
                # Save annotation
                annotation = {
                    'index': idx,
                    'date': row['date'],
                    'source': row['source'],
                    'hour': row['hour'],
                    'importance_level': row['importance_level'],
                    'headline': row['headline'],
                    'sentiment': sentiment,
                    'relevancy_politics': ratings["Politics & Government"],
                    'relevancy_economy': ratings["Economy & Finance"],
                    'relevancy_security': ratings["Security & Military"],
                    'relevancy_health': ratings["Health & Medicine"],
                    'relevancy_science': ratings["Science & Climate"],
                    'relevancy_technology': ratings["Technology"]
                }

                st.session_state.annotations.append(annotation)
                st.session_state.annotated_indices.add(idx)
                st.session_state.current_headline = None
                st.success("✅ Rating saved! Click 'Load Next Headline' to continue.")
                st.rerun()

    # Export section
    if st.session_state.annotations:
        st.markdown("---")
        st.markdown("### 💾 Export Annotations")

        output_filename = st.text_input("Output filename", "annotated_headlines.csv")

        if st.button("📥 Export to CSV", use_container_width=True):
            if save_annotations(st.session_state.annotations, output_filename):
                st.success(f"✅ Saved {len(st.session_state.annotations)} annotations to {output_filename}")
            else:
                st.error("❌ No annotations to save")

else:
    st.info("👆 Please upload your CSV file to begin annotation")
    st.markdown("""
    ### Expected CSV format:
    - `date`: Date of the headline
    - `source`: News source
    - `hour`: Hour of publication
    - `importance_level`: Importance level
    - `headline`: The headline text
    """)