#!/usr/bin/env python
# coding: utf-8

# In[7]:


#!/usr/bin/env python
# coding: utf-8

import os
import csv
import joblib
import pandas as pd
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
import streamlit as st
from datetime import datetime

class SimpleChatbot:
    def __init__(self, questions_file, label_mapping_file, model_file, type_encoder_file, customer_encoder_file):
        self.questions_file = questions_file
        self.label_mapping_file = label_mapping_file
        self.model_file = model_file
        self.type_encoder_file = type_encoder_file
        self.customer_encoder_file = customer_encoder_file
        self.load_files()
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5))
        self.question_vectors = self.vectorizer.fit_transform(self.unique_questions)
        self.spell = SpellChecker()
        self.abbreviation_dict = {"sav": "savings", "acc": "account"}

    def load_files(self):
        files_to_check = [self.questions_file, self.label_mapping_file, self.model_file, self.type_encoder_file, self.customer_encoder_file]
        for file in files_to_check:
            if not os.path.exists(file):
                raise FileNotFoundError(f"File {file} not found.")
            print(f"Loading file: {file}")  # Debugging line to verify file loading
        
        self.questions_df = pd.read_csv(self.questions_file)
        required_columns = ['Questions', 'Answers']
        for col in required_columns:
            if col not in self.questions_df.columns:
                raise KeyError(f"Column '{col}' not found in the questions file.")
        
        # Fill missing values with empty strings
        self.questions_df['Questions'] = self.questions_df['Questions'].astype(str).fillna('')
        self.questions_df['Answers'] = self.questions_df['Answers'].astype(str).fillna('')

        self.unique_questions = self.questions_df['Questions'].tolist()
        self.answers = dict(zip(self.questions_df['Questions'], self.questions_df['Answers']))
        
        self.label_mapping = pd.read_csv(self.label_mapping_file).fillna('')
        self.model = joblib.load(self.model_file)
        self.label_encoder_type = joblib.load(self.type_encoder_file)
        self.label_encoder_customer = joblib.load(self.customer_encoder_file)

        print("Files loaded successfully.")  # Debugging line to confirm files are loaded

    def preprocess_input(self, input_text):
        if input_text is None:
            return ""
        corrected_text = " ".join([self.spell.correction(word) for word in input_text.split()])
        expanded_text = " ".join([self.abbreviation_dict.get(word.lower(), word) for word in corrected_text.split()])
        return expanded_text

    def get_answer(self, question):
        preprocessed_question = self.preprocess_input(question).lower()
        print(f"Preprocessed Question: '{preprocessed_question}'")  # Debugging line
        for q in self.unique_questions:
            if q.lower() == preprocessed_question:
                return self.answers[q]
        suggestions = self.suggest_questions(preprocessed_question)
        if suggestions:
            return f"Did you mean: {', '.join(suggestions)}"
        return "Connecting to agent..."

    def suggest_questions(self, input_text, cosine_threshold=0.2, fuzzy_threshold=60):
        processed_input = self.preprocess_input(input_text)
        input_vector = self.vectorizer.transform([processed_input])
        similarities = cosine_similarity(input_vector, self.question_vectors).flatten()
        similar_indices = [idx for idx, score in enumerate(similarities) if score >= cosine_threshold]
        cosine_suggestions = [self.unique_questions[idx] for idx in similar_indices]
        fuzzy_suggestions = [q for q in self.unique_questions if fuzz.partial_ratio(processed_input, q) >= fuzzy_threshold]
        suggestions = list(dict.fromkeys(cosine_suggestions + fuzzy_suggestions))
        return suggestions[:3]  # Limit to top 3 suggestions

    def predict_next_questions(self, last_question):
        matched_row = self.label_mapping[self.label_mapping['Category'].str.lower() == last_question.lower()]
        if not matched_row.empty:
            next_questions = [
                matched_row['Next Question 1'].values[0] if matched_row['Next Question 1'].values[0] else '',
                matched_row['Next Question 2'].values[0] if matched_row['Next Question 2'].values[0] else '',
                matched_row['Next Question 3'].values[0] if matched_row['Next Question 3'].values[0] else ''
            ]
            return [q for q in next_questions if q]
        return []

def save_user_selection(typed_question, selected_question, answer):
    try:
        file_path = 'D:/ML_OPS/New/user_selections.csv'
        file_exists = os.path.isfile(file_path)
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Typed Question', 'Selected Question', 'Answer', 'Timestamp'])
            writer.writerow([typed_question, selected_question, answer, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        export_to_excel(file_path)
    except Exception as e:
        print(f"Error saving user selection: {e}")

def export_to_excel(csv_file_path):
    try:
        df = pd.read_csv(csv_file_path)
        excel_file_path = csv_file_path.replace('.csv', '.xlsx')
        df.to_excel(excel_file_path, index=False)
    except Exception as e:
        print(f"Error exporting to Excel: {e}")

questions_file = 'D:/ML_OPS/New/Q&A.csv'
label_mapping_file = 'D:/ML_OPS/New/label_mapping1.csv'
model_file = 'D:/ML_OPS/New/best_xgboost_model.pkl'
type_encoder_file = 'D:/ML_OPS/New/label_encoder_type.pkl'
customer_encoder_file = 'D:/ML_OPS/New/label_encoder_customer.pkl'

chatbot = SimpleChatbot(questions_file, label_mapping_file, model_file, type_encoder_file, customer_encoder_file)

st.title("Simple Chatbot")

st.markdown(
    """
    <style>
    .chat-bubble { max-width: 60%; padding: 10px; margin: 10px; border-radius: 15px; font-size: 16px; color: black; }
    .user { background-color: #DCF8C6; margin-left: auto; text-align: right; }
    .bot { background-color: #F1F0F0; margin-right: auto; text-align: left; }
    .chat-container { display: flex; flex-direction: column; max-height: 70vh; overflow-y: auto; }
    .suggestions-container { display: flex; flex-direction: column; margin: 10px 0; position: absolute; left: 10px; }
    .input-container { position: fixed; bottom: 0; width: 100%; background-color: white; padding: 10px; box-shadow: 0px -2px 5px rgba(0,0,0,0.1); }
    .dropdown { font-size: 14px; }
    .small-font { font-size: 12px; }
    .right-container { position: absolute; top: 80px; right: 10px; width: 20%; }
    </style>
    """,
    unsafe_allow_html=True
)

if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'suggestions' not in st.session_state:
    st.session_state.suggestions = []
if 'next_questions' not in st.session_state:
    st.session_state.next_questions = []
if 'selected_question' not in st.session_state:
    st.session_state.selected_question = None

def display_conversation():
    chat_html = '<div class="chat-container">'
    for entry in st.session_state.conversation:
        user_input, response = entry['user'], entry['bot']
        chat_html += f'<div class="chat-bubble user">{user_input}</div><div class="chat-bubble bot">{response}</div>'
    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)

def handle_suggestion_selection(selected_question):
    if selected_question and selected_question != st.session_state.selected_question:
        answer = chatbot.get_answer(selected_question)
        st.session_state.conversation.append({'user': selected_question, 'bot': answer})  # Append at bottom
        save_user_selection(selected_question, selected_question, answer)
        next_questions = chatbot.predict_next_questions(selected_question)
        st.session_state.next_questions = next_questions
        st.session_state.suggestions = []
        st.session_state.selected_question = selected_question
        st.experimental_rerun()  # Refresh the screen to show the new state

def display_suggestions():
    if st.session_state.suggestions:
        st.markdown("<div class='suggestions-container'>", unsafe_allow_html=True)
        selected_suggestion = st.radio("Did you mean:", st.session_state.suggestions, key="selectbox")
        if st.button("Select", key="select_suggestion"):
            handle_suggestion_selection(selected_suggestion)
        st.markdown("</div>", unsafe_allow_html=True)

def display_next_questions():
    if st.session_state.next_questions:
        st.markdown("<div class='small-font'>You may also want to ask:</div>", unsafe_allow_html=True)
        for question in st.session_state.next_questions:
            if question:  # Only display non-empty questions
                if st.button(question, key=f"next_{question}"):
                    handle_suggestion_selection(question)

st.button("New Question", key="new_question_top")

display_conversation()

with st.form(key='user_input_form', clear_on_submit=True):
    user_input = st.text_input("You:", key="user_input")
    submit_button = st.form_submit_button(label='Submit')
    if submit_button and user_input:
        suggestions = chatbot.suggest_questions(user_input)
        if suggestions:
            st.session_state.suggestions = suggestions
        else:
            answer = chatbot.get_answer(user_input)
            st.session_state.conversation.append({'user': user_input, 'bot': answer})  # Append at bottom
            save_user_selection(user_input, user_input, answer)
            next_questions = chatbot.predict_next_questions(user_input)
            st.session_state.next_questions = next_questions
        st.experimental_rerun()

display_suggestions()
display_next_questions()


# In[ ]:




