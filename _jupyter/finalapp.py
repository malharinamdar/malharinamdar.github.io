import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
import google.generativeai as genai

# Load the model and dataset
with open('/Users/malhar.inamdar/Desktop/streamlitapp/wowmodel2.pkl', 'rb') as file:
    saved_model = pickle.load(file)

df = pd.read_csv('/Users/malhar.inamdar/Desktop/streamlitapp/diabetes_prediction_dataset.csv')
columns = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']

# Set up GenAi model and API key
GOOGLE_API_KEY = ""
genai.configure(api_key=GOOGLE_API_KEY)

MODEL_CONFIG = {
  "temperature": 0.2,
  "top_p": 1,
  "top_k": 32,
  "max_output_tokens": 4096,
}

safety_settings = [
  {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
  {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
  {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
  {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]

model = genai.GenerativeModel('gemini-1.5-flash', safety_settings=safety_settings)

# Title and description for landing page
st.title('Welcome to DiabetesCare AI')
st.write("""
      AI Enhanced Diabetes Prediction and Gemini Driven Assistance Companion

      Take Charge of Your Health.
    """)

# Sidebar for user login and data input
with st.sidebar:
    st.header('Patient Data')
    # User login and account management
    if 'user_accounts' not in st.session_state:
        st.session_state.user_accounts = {}
    
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    
    if st.session_state.current_user is None:
        st.subheader('Access')
        user_name = st.text_input('Enter your name:')
        if st.button('Enter'):
            if user_name:
                if user_name not in st.session_state.user_accounts:
                    st.session_state.user_accounts[user_name] = {'history': []}
                st.session_state.current_user = user_name
                st.success(f"Entered as {user_name}")
            else:
                st.warning('Please enter a name.')
    else:
        st.subheader(f'Entered as {st.session_state.current_user}')
        if st.button('Exit'):
            st.session_state.current_user = None
            st.success('Exited successfully')

# Add some explanation or instructions
st.write("""
    ### Instructions
    1. Enter your name to get started.
    2. Fill in the patient data on the sidebar.
    3. Click 'Predict' to see the prediction and visualized report.
    """)

# Footer with disclaimer or additional info
st.write("""
    --- 
    Developed by Team Spam Bytes. Powered by Streamlit, Plotly, Scikit-learn, and Google Gemini.
    """)

# Data input fields
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 0, 110, 30)
hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
heart_disease = st.selectbox('Heart Disease', ['No', 'Yes'])
smoking_history = st.selectbox('Smoking History', ['current', 'non-smoker', 'past_smoker'])
height = st.number_input('Height (cm)', min_value=50, max_value=250, value=170)
weight = st.number_input('Weight (kg)', min_value=10, max_value=200, value=70)
bmi = weight / ((height / 100) ** 2)
HbA1c_level = st.slider('Haemoglobin (HbA1c) Level', 3.0, 11.0, 5.0)
blood_glucose_level = st.slider('Blood Glucose Level', 75, 310, 88)

predict_button = st.button('Predict')

# Function for user input
def user_report():
    return pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [1 if hypertension == 'Yes' else 0],
        'heart_disease': [1 if heart_disease == 'Yes' else 0],
        'smoking_history': [smoking_history],
        'bmi': [bmi],
        'HbA1c_level': [HbA1c_level],
        'blood_glucose_level': [blood_glucose_level]
    })

# Execute prediction only when button is pressed
if predict_button:
    if st.session_state.current_user:
        user_data = user_report()
        st.subheader('Patient Data')
        st.write(user_data)

        # Prediction function
        def dia_predict(adata):
            return saved_model.predict(adata)

        user_result = dia_predict(user_data)

        # Visualization section
        st.title('Visualized Patient Report')

        # COLOR FUNCTION
        color = 'blue' if user_result[0] == 0 else 'darkred'

        # Age vs Glucose
        with st.expander('Age vs Glucose'):
            st.header('Age vs Glucose')
            fig_glucose = plt.figure()
            sns.scatterplot(x='age', y='blood_glucose_level', data=df, hue='diabetes', color=color)
            sns.scatterplot(x=user_data['age'], y=user_data['blood_glucose_level'], s=150, color=color)
            plt.xticks(np.arange(10, 100, 5))
            plt.yticks(np.arange(70, 310, 20))
            plt.title('0 - Healthy & 1 - Unhealthy')
            st.pyplot(fig_glucose)
            st.write("This graph shows the relationship between age and blood glucose levels in the dataset. Each point represents a patient's age and their corresponding blood glucose level, with color indicating if the patient has diabetes or not. The highlighted point is the input patient's data.")

        # Age vs Heart Disease
        with st.expander('Age vs Heart Disease'):
            st.header('Age vs Heart Disease')
            fig_bp = plt.figure()
            sns.scatterplot(x='age', y='heart_disease', data=df, hue='diabetes', color=color)
            sns.scatterplot(x=user_data['age'], y=user_data['heart_disease'], s=150, color=color)
            plt.xticks(np.arange(0, 110, 10))
            plt.yticks(np.arange(0, 2, 1))
            plt.title('0 - Healthy & 1 - Unhealthy')
            st.pyplot(fig_bp)
            st.write("This graph depicts the relationship between age and heart disease presence in the dataset. Each point shows a patient's age and heart disease status, with colors indicating diabetes presence. The highlighted point is the input patient's data.")

        # Age vs Smoking History
        with st.expander('Age vs Smoking History'):
            st.header('Age vs Smoking History')
            fig_st = plt.figure()
            sns.scatterplot(x='age', y='smoking_history', data=df, hue='diabetes', color=color)
            sns.scatterplot(x=user_data['age'], y=user_data['smoking_history'], s=150, color=color)
            plt.xticks(np.arange(10, 100, 5))
            plt.yticks(np.arange(0, 4, 1))
            plt.title('0 - Healthy & 1 - Unhealthy')
            st.pyplot(fig_st)
            st.write("This graph illustrates the relationship between age and smoking history. Each point represents a patient's age and smoking history, with colors indicating the presence of diabetes. The highlighted point is the input patient's data.")

        # Age vs HbA1c Level
        with st.expander('Age vs Haemoglobin (HbA1c) Level'):
            st.header('Age vs Haemoglobin (HbA1c) Level')
            fig_i = plt.figure()
            sns.scatterplot(x='age', y='HbA1c_level', data=df, hue='diabetes', color=color)
            sns.scatterplot(x=user_data['age'], y=user_data['HbA1c_level'], s=150, color=color)
            plt.xticks(np.arange(10, 100, 5))
            plt.yticks(np.arange(2.0, 10.0, 0.5))
            plt.title('0 - Healthy & 1 - Unhealthy')
            st.pyplot(fig_i)
            st.write("This graph presents the relationship between age and HbA1c levels. Each point represents a patient's age and HbA1c level, with colors indicating the presence of diabetes. The highlighted point is the input patient's data.")

        # Age vs BMI
        with st.expander('Age vs BMI'):
            st.header('Age vs BMI')
            fig_bmi = plt.figure()
            sns.scatterplot(x='age', y='bmi', data=df, hue='diabetes', color=color)
            sns.scatterplot(x=user_data['age'], y=user_data['bmi'], s=150, color=color)
            plt.xticks(np.arange(10, 100, 5))
            plt.yticks(np.arange(15, 60, 5))
            plt.title('0 - Healthy & 1 - Unhealthy')
            st.pyplot(fig_bmi)
            st.write("This graph displays the relationship between age and BMI. Each point represents a patient's age and BMI, with colors indicating the presence of diabetes. The highlighted point is the input patient's data.")

        # Age vs Hypertension
        with st.expander('Age vs Hypertension'):
            st.header('Age vs Hypertension')
            fig_dpf = plt.figure()
            sns.scatterplot(x='age', y='hypertension', data=df, hue='diabetes', color=color)
            sns.scatterplot(x=user_data['age'], y=user_data['hypertension'], s=150, color=color)
            plt.xticks(np.arange(10, 100, 5))
            plt.yticks(np.arange(0, 2, 1))
            plt.title('0 - Healthy & 1 - Unhealthy')
            st.pyplot(fig_dpf)
            st.write("This graph shows the relationship between age and hypertension. Each point represents a patient's age and hypertension status, with colors indicating the presence of diabetes. The highlighted point is the input patient's data.")

        # Output result
        st.subheader('Your Report:')
        output = 'You are not Diabetic' if user_result[0] == 0 else 'You are Diabetic'
        st.markdown(f"<h1 style='color: {'green' if user_result[0] == 0 else 'red'};'>{output}</h1>", unsafe_allow_html=True)

        # Suggestions from Gemini API
        def generate_suggestion(report_data, user_result):
            input_text = ", ".join([f"{col}: {val}" for col, val in zip(columns, report_data.iloc[0])])
            if user_result[0] == 0:
                return "You are not diabetic but still keep a healthy lifestyle to prevent future diagnosis."
            else:
                input_string = f"Give me personalised lifestyle and dietary suggestions for a patient with diabetes as per the data given: {input_text}, also give me helpline numbers and valuable internet resources about hospitals with good diabetic care in india"
                try:
                    response = model.generate_content(input_string)  # Use the generate_content method
                    return response.text
                except Exception as e:
                    return f"Error generating suggestion: {str(e)}"

        suggestion = generate_suggestion(user_data, user_result)
        st.subheader('Suggestions:')
        st.write(suggestion)

        # Store history
        if 'history' not in st.session_state.user_accounts[st.session_state.current_user]:
            st.session_state.user_accounts[st.session_state.current_user]['history'] = []
        st.session_state.user_accounts[st.session_state.current_user]['history'].append({
            **user_data.to_dict(orient='records')[0],
            'prediction': output
        })

        st.subheader('Prediction History')
        history = st.session_state.user_accounts[st.session_state.current_user]['history']
        if history:
            # Create a DataFrame to display history
            history_df = pd.DataFrame(history)
            # Display as a table
            st.table(history_df)
        else:
            st.write("No history found.")
    else:
        st.warning('Please enter your name before making predictions.') 

#chatbot

# Initialize the chat object
chat = model.start_chat(history=[])

def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response

st.header('Q & A Chatbot')

# Ensure chat_history is defined in session state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

input_text = st.text_input("Input: ", key="input")
submit = st.button("Ask the question")

if submit and input_text:
    response_chunks = get_gemini_response(input_text)
    # Concatenate the bot response chunks into a single message
    bot_response = ''.join([chunk.text for chunk in response_chunks])
    # Add user query and response to session state chat history
    st.session_state['chat_history'].append(("You", input_text))
    st.session_state['chat_history'].append(("Bot", bot_response))
    st.subheader("The Response is")
    st.write(bot_response)
    
    st.subheader("The Chat History is")
    for role, text in st.session_state['chat_history']:
        if role == "You":
            st.markdown(f'<div class="user-msg"><b>{role}:</b> {text}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-msg"><b>{role}:</b> {text}</div>', unsafe_allow_html=True)

# Add CSS styles
st.markdown("""
    <style>
    .user-msg {
        background-color: #e1f5fe;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .bot-msg {
        background-color: #ffe6e6;  /* Lightest shade of pink */
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)
