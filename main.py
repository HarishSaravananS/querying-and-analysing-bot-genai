import streamlit as st
import pandas as pd
import os
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory  # Importing memory module
import random


st.set_page_config(page_title="CSV Multi-Agent", layout="wide")
st.title(" CSV Multi-Agent Assistant (Groq + LangChain)")

csv_file = st.file_uploader(" Upload a CSV file", type=["csv"])

groq_key = st.text_input(" Enter your Groq API Key", type="password")

if csv_file and groq_key:
    df = pd.read_csv(csv_file)


    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

   
    llm = ChatGroq(api_key=groq_key, model_name="llama-3.3-70b-versatile")

    memory = ConversationBufferMemory(memory_key="conversation_history", return_messages=True)

    # Tool 1: QA Tool
    @tool
    def qa_tool(query: str) -> str:
        """Answer questions about the CSV."""
        try:
            sample = df.head(3).to_string()
            return f"Here's a sample of the data:\n{sample}\nAsk more specific questions."
        except Exception as e:
            return f"Error in qa_tool: {e}"

    # Tool 2: Recommender / Summary Tool
    @tool
    def analysis_tool_agent(task: str) -> str:
        """Recommend or summarize CSV content."""
        try:
            if "recommend" in task.lower():
                top = df.groupby("product")["sales"].sum().sort_values(ascending=False).head(3)
                return f" Top products:\n{top.to_string()}"
            elif "summary" in task.lower():
                return f"🧾 Summary:\n{df.describe().to_string()}"
            else:
                return "Try asking for 'recommend' or 'summary'."
        except Exception as e:
            return f"Error in analysis_tool_agent: {e}"
    @tool
    def friendly_answer_tool(query: str) -> str:
        """Provide a friendly, conversational response."""
        friendly_responses = [
        "Hey there!  How can I help you today?",
        "Hi!  I’d be happy to assist you with anything!",
        "Hello!  What can I do for you today?",
        "Hey! I'm here to help.  Feel free to ask me anything!",
        "Hi there!  How can I assist you today?"]
        return random.choice(friendly_responses)

    multi_agent = initialize_agent(
        tools=[qa_tool, analysis_tool_agent,friendly_answer_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory, 
        handle_parsing_errors=True,
    )

   
    st.subheader("💬 Chat with the Agent")
    user_input = st.text_input("Ask a question about your data:")
    if user_input:
        with st.spinner("Thinking..."):
            response = multi_agent.run(user_input)
            st.success(response)

else:
    st.warning("Please upload a CSV file and enter your Groq API key.")
