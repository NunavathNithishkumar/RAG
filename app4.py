import streamlit as st
from transformers import pipeline
import PyPDF2
import os

# Initialize the Question Answering pipeline with a larger model
qa_pipeline = pipeline("question-answering", model="deepset/bert-large-uncased-whole-word-masking-squad2")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with open(pdf_file, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

# Function to upload PDF file and ask questions
def upload_pdf_and_ask_questions(uploaded_file, question):
    # Save the uploaded file to a temporary location
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract text from the PDF
    text = extract_text_from_pdf("temp.pdf")

    # Use the QA pipeline to answer the question
    answer = qa_pipeline({
        'context': text,
        'question': question
    })

    # Delete the temporary file
    os.remove("temp.pdf")

    return answer['answer']

# Streamlit app
def main():
    st.title("PDF Question Answering System")

    # File upload
    uploaded_file = st.file_uploader("Upload PDF file", type=['pdf'])

    # Ask question
    question = st.text_input("Ask your question here")

    if uploaded_file is not None and question:
        # Convert uploaded file to string and display file details
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
        st.write(file_details)

        # Get answer
        answer = upload_pdf_and_ask_questions(uploaded_file, question)
        st.write("Question:", question)
        st.write("Answer:", answer)

if __name__ == '__main__':
    main()


