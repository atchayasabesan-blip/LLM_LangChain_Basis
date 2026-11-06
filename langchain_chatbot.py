
import pandas as pd
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==========================================
# 1. PROJECT SETUP
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
DATASET_DIR.mkdir(exist_ok=True)

llm = ChatOllama(
    model="llama3.2:latest",
    temperature=0.3
)

parser = StrOutputParser()

print("LangChain Education Chatbot Started!")


# ==========================================
# 2. FIVE PROMPT TEMPLATES
# ==========================================

prompts = {

    "General Education": PromptTemplate.from_template(
        """You are an Education Assistant.
        Answer in simple English.
        Question: {question}
        Answer:"""
    ),

    "Concept Explanation": PromptTemplate.from_template(
        """You are a teacher.
        Explain the concept with definition and example.
        Concept: {question}
        Explanation:"""
    ),

    "Exam Preparation": PromptTemplate.from_template(
        """You are an exam preparation assistant.
        Give important points and a short conclusion.
        Question: {question}
        Answer:"""
    ),

    "Programming Assistance": PromptTemplate.from_template(
        """You are a programming tutor.
        Explain step by step with a code example.
        Question: {question}
        Solution:"""
    ),

    "Career Guidance": PromptTemplate.from_template(
        """You are a career guide.
        Give useful skills and learning steps.
        Question: {question}
        Guidance:"""
    )
}

prompt_data = []

for name, template in prompts.items():
    print("\nPROMPT:", name)

    prompt_data.append({
        "PROMPT_NAME": name,
        "PROMPT_TEMPLATE": template.template
    })

pd.DataFrame(prompt_data).to_csv(
    DATASET_DIR / "prompts.csv",
    index=False
)


# ==========================================
# 3. EDUCATION CHATBOT CHAIN
# ==========================================

education_chain = (
    prompts["General Education"]
    | llm
    | parser
)


# ==========================================
# 4. TEST WITH 20 USER QUERIES
# ==========================================

user_queries = [
    "What is Artificial Intelligence?",
    "Explain Machine Learning.",
    "What is Deep Learning?",
    "Explain Neural Networks.",
    "What is Python?",
    "Explain Python variables.",
    "What is DBMS?",
    "Explain SQL.",
    "What is a Primary Key?",
    "What is Cloud Computing?",
    "Explain Data Science.",
    "What is an Algorithm?",
    "Explain Data Structures.",
    "What is Computer Networking?",
    "Explain Cybersecurity.",
    "What is NLP?",
    "Explain Transformer Architecture.",
    "What is Generative AI?",
    "How to prepare for programming exams?",
    "What skills are required for an AI career?"
]

response_data = []

for i, question in enumerate(user_queries, start=1):

    print("\nQUERY", i, ":", question)

    response = education_chain.invoke({
        "question": question
    })

    print("RESPONSE:", response)

    response_data.append({
        "QUERY_ID": i,
        "USER_QUERY": question,
        "CHATBOT_RESPONSE": response
    })

pd.DataFrame(response_data).to_csv(
    DATASET_DIR / "chatbot_responses.csv",
    index=False
)

print("\n20 Queries Completed!")


# ==========================================
# 5. PROMPT MODIFICATION AND COMPARISON
# ==========================================

question = "Explain Artificial Intelligence."

original_chain = (
    prompts["General Education"]
    | llm
    | parser
)

original_response = original_chain.invoke({
    "question": question
})

improved_prompt = PromptTemplate.from_template(
    """You are an experienced college teacher.

    Explain the topic in simple English.

    Include:
    1. Definition
    2. Important points
    3. Real-world example
    4. Short conclusion

    Topic: {question}

    Answer:"""
)

improved_chain = improved_prompt | llm | parser

improved_response = improved_chain.invoke({
    "question": question
})

print("\nORIGINAL PROMPT RESPONSE:")
print(original_response)

print("\nIMPROVED PROMPT RESPONSE:")
print(improved_response)

comparison_data = [
    {
        "PROMPT_VERSION": "Original",
        "QUESTION": question,
        "RESPONSE": original_response
    },
    {
        "PROMPT_VERSION": "Improved",
        "QUESTION": question,
        "RESPONSE": improved_response
    }
]

pd.DataFrame(comparison_data).to_csv(
    DATASET_DIR / "prompt_comparison.csv",
    index=False
)


# ==========================================
# 6. SUMMARIZATION CHAIN
# ==========================================

summary_prompt = PromptTemplate.from_template(
    """You are a summarization assistant.

    Summarize the following text in simple English.
    Include only important points.

    Text: {text}

    Summary:"""
)

summarization_chain = (
    summary_prompt
    | llm
    | parser
)

sample_text = """
Artificial Intelligence is a branch of computer science.
It enables machines to perform tasks such as learning,
reasoning and problem-solving.

Machine Learning is a subset of Artificial Intelligence.
It allows computers to learn from data.

AI is widely used in education, healthcare,
banking and many other industries.
"""

summary_result = summarization_chain.invoke({
    "text": sample_text
})

print("\n========== SUMMARIZATION CHAIN ==========")
print("ORIGINAL TEXT:")
print(sample_text)

print("\nGENERATED SUMMARY:")
print(summary_result)


# ==========================================
# 7. QUESTION ANSWERING CHAIN
# ==========================================

qa_prompt = PromptTemplate.from_template(
    """You are an educational question-answering assistant.

    Answer using only the given context.
    If the answer is unavailable, say so.

    Context: {context}

    Question: {question}

    Answer:"""
)

qa_chain = qa_prompt | llm | parser

context = """
Python is a high-level programming language.
It is easy to learn and supports multiple programming
paradigms. Python is widely used in AI, data science,
web development and automation.
"""

qa_question = "What are the uses of Python?"

qa_result = qa_chain.invoke({
    "context": context,
    "question": qa_question
})

print("\n========== QUESTION ANSWERING CHAIN ==========")
print("CONTEXT:")
print(context)

print("\nQUESTION:")
print(qa_question)

print("\nANSWER:")
print(qa_result)


# ==========================================
# 8. SAVE CHAIN RESULTS
# ==========================================

chain_results = [
    {
        "CHAIN_NAME": "Summarization Chain",
        "INPUT": sample_text,
        "OUTPUT": summary_result
    },
    {
        "CHAIN_NAME": "Question Answering Chain",
        "INPUT": context,
        "OUTPUT": qa_result
    }
]

pd.DataFrame(chain_results).to_csv(
    DATASET_DIR / "chain_results.csv",
    index=False
)


# ==========================================
# 9. FINAL RESULT
# ==========================================

print("\n================================")
print("WEEK 10 LANGCHAIN PROJECT COMPLETED")
print("================================")

print("\nGenerated CSV Files:")

for file in DATASET_DIR.glob("*.csv"):
    print(file.name)

print("\nAll tasks completed successfully!")
