import streamlit as st
import joblib
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and preprocessing objects
model = joblib.load('soft_skills_model.pkl')
encoder = joblib.load('encoder.pkl')

# Load dataset for visualizations
data = pd.read_csv('soft_skills_dataset.csv')  # Replace with your actual dataset

# Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# HTML Header
def html_header():
    st.markdown("""
    <div class="header" style="text-align:center; padding:2rem; background:#2d3436; border-radius:15px; margin-bottom:2rem;">
        <h1 style="color:#4ecdc4; margin:0;">🎯 Soft Skills Analyzer</h1>
        <p style="color:#dfe6e9;">Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

# HTML Footer
def html_footer():
    st.markdown("""
    <div style="text-align:center; margin-top:3rem; padding:1rem; background:#2d3436; color:white; border-radius:15px;">
        <p>© 2024 Skill Prediction System | Contact: support@skills.ai</p>
    </div>
    """, unsafe_allow_html=True)

    
def main():
    # Load CSS
    local_css("static/styles.css")
    
    # Add HTML Header
    html_header()

    # Data Overview Section
    with st.expander("📊 See Data Overview"):
        st.write("### Dataset Statistics")
        st.dataframe(data.describe())
        
        # Filter numeric columns
        numeric_data = data.select_dtypes(include=['number'])

        # Correlation Heatmap
        st.write("### Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # Input Section
    with st.form("prediction_form"):
        st.header("📝 Input Your Details")
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 18, 65, 25)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            education = st.selectbox("Education Level", 
                                   ["High School", "Bachelor's", "Master's", "PhD"])
            
        with col2:
            profession = st.selectbox("Profession", 
                                    ["Manager", "Software Engineer", "Data Scientist", "Other"])
            skill = st.selectbox("Skill Category", 
                               ["Communication", "Leadership", "Technical", "Other"])
            
        initial_skill = st.slider("Initial Skill Level", 0.0, 10.0, 5.0, step=0.5)
        training_hours = st.number_input("Training Hours Completed", 0, 500, 50)
        confidence = st.slider("Confidence Level", 0, 10, 5)
        feedback = st.slider("Feedback Score", 0, 10, 5)
        activity = st.slider("Activity Participation", 0, 10, 5)

        submitted = st.form_submit_button("🚀 Predict My Skill Level")

    if submitted:
        # Process inputs
        input_data = pd.DataFrame([{
            "Gender": gender,
            "Education Level": education,
            "Profession": profession,
            "Skill Category": skill,
            "Age": age,
            "Initial Skill Level": initial_skill,
            "Training Hours": training_hours,
            "Confidence Level": confidence,
            "Feedback Score": feedback,
            "Activity Participation": activity
        }])

        # Preprocess data
        encoded_cat = encoder.transform(input_data[["Gender", "Education Level", "Profession", "Skill Category"]])
        encoded_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out())
        numerical_data = input_data[["Age", "Initial Skill Level", "Training Hours", 
                                   "Confidence Level", "Feedback Score", "Activity Participation"]]
        final_input = pd.concat([numerical_data, encoded_df], axis=1)

        # Get the model's expected feature names
        expected_columns = model.feature_names_in_

        # Add missing columns with default values (0)
        for col in expected_columns:
            if col not in final_input.columns:
                final_input[col] = 0

        # Reorder columns to match the model's expected input
        final_input = final_input[expected_columns]

        # Make prediction
        prediction = model.predict(final_input)[0]
        improvement = prediction - initial_skill

        # Display Results
        st.success("📈 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Skill Level", f"{prediction:.2f}/10")
        with col2:
            st.metric("Skill Improvement", f"+{improvement:.2f}" if improvement > 0 else f"{improvement:.2f}")
        with col3:
            st.metric("Recommended Training", f"{int(training_hours*1.2)} hours")

        # Progress Visualization
        st.write("### Skill Development Progress")
        fig, ax = plt.subplots()
        ax.barh(['Initial', 'Current'], [initial_skill, prediction], color=['#ff6b6b', '#4ecdc4'])
        ax.set_xlim(0, 10)
        st.pyplot(fig)

        # Feature Importance (if available)
        try:
            st.write("### Key Factors Influencing Your Skill Level")
            importance = model.feature_importances_
            features = final_input.columns
            imp_df = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values('Importance', ascending=False).head(5)
            
            fig, ax = plt.subplots()
            sns.barplot(x='Importance', y='Feature', data=imp_df, palette='viridis', ax=ax)
            st.pyplot(fig)
        except:
            st.info("Feature importance not available for this model type")

        # Training Recommendations
        st.write("### 📚 Personalized Recommendations")
        if improvement > 2:
            st.success("🌟 Outstanding progress! Keep up the good work with:")
        else:
            st.warning("💡 Potential for improvement. Focus on:")
            
        if skill == "Communication":
            st.markdown("- Practice public speaking exercises\n- Join debate clubs\n- Take active listening courses")
        elif skill == "Technical":
            st.markdown("- Complete coding challenges\n- Attend workshops\n- Participate in hackathons")

    # Add HTML Footer
    html_footer()

 
 
 # This should be OUTSIDE the main() function
if __name__ == "__main__":
    main()
    
    
    