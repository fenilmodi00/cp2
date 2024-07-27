import streamlit as st
from transformers import pipeline

# Load the model from Hugging Face
model_name = "microsoft/Phi-3-mini-4k-instruct"
model = pipeline("text-generation", model=model_name)

# Define the Streamlit form
st.title("Personalized Fitness Recommendation App")

with st.form("fitness_form"):
    age = st.number_input("Age", min_value=1, max_value=100, value=25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
    weight = st.number_input("Weight (kg)", min_value=10, max_value=300, value=70)
    activity_level = st.selectbox("Activity Level", ["Sedentary", "Lightly active", "Moderately active", "Very active", "Extra active"])
    fitness_goal = st.selectbox("Fitness Goal", ["Lose weight", "Maintain weight", "Gain muscle", "Improve fitness"])
    submit = st.form_submit_button("Get Recommendation")

# Generate recommendation on form submission
if submit:
    # Custom prompt for the model
    prompt = (f"Provide a personalized fitness training schedule for a {age}-year-old {gender} "
              f"who is {height} cm tall, weighs {weight} kg, has an activity level of {activity_level}, "
              f"and has a fitness goal to {fitness_goal}. Include recommended exercises. Also include a sample diet which he/she should have.")

    # Generate the recommendation
    result = model(prompt, max_length=250, num_return_sequences=1)[0]["generated_text"]
    st.write("### Personalized Fitness Training Schedule")
    st.write(result)