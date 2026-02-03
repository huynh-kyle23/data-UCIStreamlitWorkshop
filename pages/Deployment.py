import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from pathlib import Path

st.set_page_config(
    page_title="Toxic Comment Classifier",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Toxic Comment Classification Model")
st.markdown("### Classify text comments for toxicity using Logistic Regression")

# Cache the model loading to avoid reloading on every interaction
@st.cache_resource
def load_models():
    """Load the trained models and vectorizer"""
    try:
        # Try to load the bundle first
        bundle_path = Path('models/logistic_regression_bundle.pkl')
        if bundle_path.exists():
            bundle = joblib.load(bundle_path)
            return bundle['vectorizer'], bundle['models'], bundle['labels'], bundle['hyperparameters']
        else:
            st.error(f"Model bundle not found at {bundle_path}")
            st.info("Please ensure you've run the training script and saved the models.")
            return None, None, None, None
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

# Load models
with st.spinner("Loading models..."):
    vectorizer, models, labels, hyperparameters = load_models()

if vectorizer is None or models is None:
    st.stop()

# Display model info in sidebar
st.header("📊 Model Information")
st.markdown(f"**Algorithm:** Logistic Regression")
st.markdown(f"**Max Features:** {hyperparameters['max_features']:,}")
st.markdown(f"**Regularization (C):** {hyperparameters['C']}")
st.markdown(f"**Labels:** {len(labels)}")

# Load best results if available
try:
    with open('Evals/logistic_regression_tuning_best.json', 'r') as f:
        import json
        best_results = json.load(f)
        st.markdown("---")
        st.markdown("**Performance Metrics:**")
        st.markdown(f"- Macro F1: {best_results['macro_f1']}")
        st.markdown(f"- Macro Precision: {best_results['macro_precision']}")
        st.markdown(f"- Macro Recall: {best_results['macro_recall']}")
        st.markdown(f"- Macro AUC-PR: {best_results['macro_auc_pr']}")
except:
    pass

st.markdown("---")
st.markdown("**Label Definitions:**")
st.markdown("- **Toxic:** Rude, disrespectful, or unreasonable")
st.markdown("- **Severe Toxic:** Very hateful, aggressive, or disrespectful")
st.markdown("- **Obscene:** Contains swear words, curse words, or profanity")
st.markdown("- **Threat:** Describes intent of inflicting pain, injury, or violence")
st.markdown("- **Insult:** Insulting, inflammatory, or negative comment")
st.markdown("- **Identity Hate:** Contains hate speech based on identity")

# Main content
st.markdown("---")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📝 Batch Prediction", "📈 Model Info"])

with tab1:
    st.header("Single Comment Classification")
    st.markdown("Enter a comment below to classify it for various types of toxicity.")
    
    # Text input
    comment = st.text_area(
        "Enter comment to classify:",
        placeholder="Type or paste a comment here...",
        height=150,
        key="single_comment"
    )
    
    # Prediction threshold slider
    threshold = st.slider(
        "Prediction Threshold (probability above which a label is considered positive):",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        key="threshold"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        predict_button = st.button("🔍 Classify", type="primary", width="stretch")
    
    if predict_button and comment.strip():
        with st.spinner("Classifying comment..."):
            # Transform the comment
            X_comment = vectorizer.transform([comment])
            
            # Get predictions from all models
            predictions = {}
            probabilities = {}
            
            for label in labels:
                pred = models[label].predict(X_comment)[0]
                prob = models[label].predict_proba(X_comment)[0, 1]
                predictions[label] = pred
                probabilities[label] = prob
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Classification Results")
            
            # Overall toxicity check
            any_toxic = any(probabilities[label] >= threshold for label in labels)
            
            if any_toxic:
                st.error("⚠️ This comment may contain toxic content!")
            else:
                st.success("✅ This comment appears to be non-toxic.")
            
            # Create results dataframe
            results_df = pd.DataFrame({
                'Label': labels,
                'Probability': [probabilities[label] for label in labels],
                'Prediction': ['Positive' if probabilities[label] >= threshold else 'Negative' for label in labels]
            })
            results_df = results_df.sort_values('Probability', ascending=False)
            
            # Display as metrics
            st.markdown("#### Detailed Predictions:")
            cols = st.columns(3)
            for idx, (_, row) in enumerate(results_df.iterrows()):
                with cols[idx % 3]:
                    label_name = row['Label'].replace('_', ' ').title()
                    prob_pct = row['Probability'] * 100
                    
                    if row['Prediction'] == 'Positive':
                        st.metric(
                            label=f"🔴 {label_name}",
                            value=f"{prob_pct:.1f}%",
                            delta="Detected"
                        )
                    else:
                        st.metric(
                            label=f"🟢 {label_name}",
                            value=f"{prob_pct:.1f}%",
                            delta="Not Detected"
                        )
            
            # Show probability bar chart
            st.markdown("#### Probability Scores:")
            results_df_chart = results_df.set_index('Label')
            st.bar_chart(results_df_chart['Probability'])
            
            # Show detailed table
            with st.expander("View Detailed Results Table"):
                results_df['Probability'] = results_df['Probability'].apply(lambda x: f"{x:.4f}")
                st.dataframe(results_df, width="stretch", hide_index=True)
    
    elif predict_button:
        st.warning("⚠️ Please enter a comment to classify.")
    
    # Example comments
    st.markdown("---")
    st.markdown("**Try these example comments:**")
    
    example_cols = st.columns(3)
    with example_cols[0]:
        if st.button("Example: Clean Comment", width="stretch"):
            st.session_state.single_comment = "This is a great discussion. Thanks for sharing your thoughts!"
            st.rerun()
    
    with example_cols[1]:
        if st.button("Example: Mildly Toxic", width="stretch"):
            st.session_state.single_comment = "That's a stupid idea and you don't know what you're talking about."
            st.rerun()
    
    with example_cols[2]:
        if st.button("Example: Highly Toxic", width="stretch"):
            st.session_state.single_comment = "You're an idiot and everyone hates you. Get lost!"
            st.rerun()

with tab2:
    st.header("Batch Comment Classification")
    st.markdown("Upload a CSV file with comments to classify multiple comments at once.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload CSV file (must contain a 'comment_text' column):",
        type=['csv'],
        help="Upload a CSV file with a column named 'comment_text' containing the comments to classify."
    )
    
    batch_threshold = st.slider(
        "Prediction Threshold for Batch:",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        key="batch_threshold"
    )
    
    if uploaded_file is not None:
        try:
            # Read the CSV
            df = pd.read_csv(uploaded_file)
            
            if 'comment_text' not in df.columns:
                st.error("❌ CSV must contain a 'comment_text' column!")
            else:
                st.success(f"✅ Loaded {len(df)} comments from file")
                
                # Show preview
                with st.expander("Preview uploaded data"):
                    st.dataframe(df.head(10), width="stretch")
                
                if st.button("🚀 Classify All Comments", type="primary"):
                    with st.spinner(f"Classifying {len(df)} comments..."):
                        # Transform all comments
                        X_batch = vectorizer.transform(df['comment_text'])
                        
                        # Get predictions for each label
                        progress_bar = st.progress(0)
                        for idx, label in enumerate(labels):
                            df[f'{label}_prob'] = models[label].predict_proba(X_batch)[:, 1]
                            df[f'{label}_pred'] = (df[f'{label}_prob'] >= batch_threshold).astype(int)
                            progress_bar.progress((idx + 1) / len(labels))
                        
                        # Calculate overall toxicity
                        prob_cols = [f'{label}_prob' for label in labels]
                        df['max_toxicity_prob'] = df[prob_cols].max(axis=1)
                        df['is_toxic'] = (df['max_toxicity_prob'] >= batch_threshold).astype(int)
                        df['num_toxic_labels'] = df[[f'{label}_pred' for label in labels]].sum(axis=1)
                        
                        st.success("✅ Classification complete!")
                        
                        # Summary statistics
                        st.markdown("### 📊 Summary Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Comments", len(df))
                        with col2:
                            toxic_count = df['is_toxic'].sum()
                            toxic_pct = (toxic_count / len(df)) * 100
                            st.metric("Toxic Comments", f"{toxic_count} ({toxic_pct:.1f}%)")
                        with col3:
                            clean_count = len(df) - toxic_count
                            clean_pct = (clean_count / len(df)) * 100
                            st.metric("Clean Comments", f"{clean_count} ({clean_pct:.1f}%)")
                        with col4:
                            avg_labels = df['num_toxic_labels'].mean()
                            st.metric("Avg Toxic Labels", f"{avg_labels:.2f}")
                        
                        # Label distribution
                        st.markdown("### 📈 Label Distribution")
                        label_counts = {label: df[f'{label}_pred'].sum() for label in labels}
                        label_df = pd.DataFrame({
                            'Label': list(label_counts.keys()),
                            'Count': list(label_counts.values()),
                            'Percentage': [v / len(df) * 100 for v in label_counts.values()]
                        })
                        label_df = label_df.sort_values('Count', ascending=False)
                        
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.bar_chart(label_df.set_index('Label')['Count'])
                        with col2:
                            st.dataframe(
                                label_df.style.format({'Percentage': '{:.1f}%'}),
                                width="stretch",
                                hide_index=True
                            )
                        
                        # Show results
                        st.markdown("### 📋 Detailed Results")
                        
                        # Filter options
                        filter_col1, filter_col2 = st.columns(2)
                        with filter_col1:
                            show_filter = st.selectbox(
                                "Filter by:",
                                ["All Comments", "Toxic Only", "Clean Only"]
                            )
                        
                        with filter_col2:
                            sort_by = st.selectbox(
                                "Sort by:",
                                ["Original Order", "Max Toxicity (High to Low)", "Max Toxicity (Low to High)"]
                            )
                        
                        # Apply filters
                        display_df = df.copy()
                        if show_filter == "Toxic Only":
                            display_df = display_df[display_df['is_toxic'] == 1]
                        elif show_filter == "Clean Only":
                            display_df = display_df[display_df['is_toxic'] == 0]
                        
                        # Apply sorting
                        if sort_by == "Max Toxicity (High to Low)":
                            display_df = display_df.sort_values('max_toxicity_prob', ascending=False)
                        elif sort_by == "Max Toxicity (Low to High)":
                            display_df = display_df.sort_values('max_toxicity_prob', ascending=True)
                        
                        st.dataframe(display_df, width="stretch", hide_index=True)
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="toxic_classification_results.csv",
                            mime="text/csv",
                            type="primary"
                        )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    else:
        st.info("👆 Upload a CSV file to get started with batch classification")
        
        # Sample CSV template
        st.markdown("---")
        st.markdown("**Need a template?** Download this sample CSV to see the expected format:")
        sample_df = pd.DataFrame({
            'comment_text': [
                'This is a great discussion!',
                'You are an idiot.',
                'Thanks for sharing your perspective.',
                'This is absolutely terrible and you should be ashamed.'
            ]
        })
        sample_csv = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=sample_csv,
            file_name="sample_comments.csv",
            mime="text/csv"
        )

with tab3:
    st.header("📈 Model Information & Performance")
    
    # Load and display best results
    try:
        with open('../results/logistic_regression_tuning_best.json', 'r') as f:
            import json
            best_results = json.load(f)
        
        st.markdown("### Overall Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Macro F1 Score", f"{best_results['macro_f1']:.3f}")
        with col2:
            st.metric("Macro Precision", f"{best_results['macro_precision']:.3f}")
        with col3:
            st.metric("Macro Recall", f"{best_results['macro_recall']:.3f}")
        with col4:
            st.metric("Macro AUC-PR", f"{best_results['macro_auc_pr']:.3f}")
        
        # Per-label performance
        st.markdown("### Per-Label AUC-PR Scores")
        per_label_df = pd.DataFrame({
            'Label': list(best_results['per_label_auc_pr'].keys()),
            'AUC-PR': list(best_results['per_label_auc_pr'].values())
        })
        per_label_df = per_label_df.sort_values('AUC-PR', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.bar_chart(per_label_df.set_index('Label')['AUC-PR'])
        with col2:
            st.dataframe(
                per_label_df.style.format({'AUC-PR': '{:.3f}'}),
                width="stretch",
                hide_index=True
            )
        
        # Model architecture
        st.markdown("### Model Architecture")
        st.markdown(f"""
        - **Algorithm:** Logistic Regression with L2 regularization
        - **Vectorization:** TF-IDF with {hyperparameters['max_features']:,} max features
        - **N-gram Range:** (1, 2) - unigrams and bigrams
        - **Class Weighting:** Balanced
        - **Regularization Parameter (C):** {hyperparameters['C']}
        - **Multi-label Strategy:** One model per label (6 independent classifiers)
        - **Cross-validation:** 4-fold stratified
        """)
        
        # Display tuning plot if available
        plot_path = Path('../Evals/logistic_regression_tuning_plot.png')
        if plot_path.exists():
            st.markdown("### Hyperparameter Tuning Results")
            st.image(str(plot_path), width="stretch")
        
    except FileNotFoundError:
        st.warning("⚠️ Performance metrics file not found. Please run the training script first.")
    except Exception as e:
        st.error(f"❌ Error loading performance metrics: {str(e)}")
    
    # About section
    st.markdown("---")
    st.markdown("### About This Application")
    st.markdown("""
    This application uses machine learning to classify text comments for various types of toxicity.
    The model was trained on labeled data and uses TF-IDF features with Logistic Regression classifiers.
    
    **Use Cases:**
    - Content moderation for online platforms
    - Automated comment filtering
    - Toxicity detection in user-generated content
    - Research on online discourse
    
    **Note:** This is a machine learning model and may not be 100% accurate. 
    Human review is recommended for critical moderation decisions.
    """)

st.markdown("---")
st.markdown("Built with Streamlit • Powered by scikit-learn")