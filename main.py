# app.py

# Import Dependencies
import os
import certifi
import requests
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.tools import tool
from langchain.agents import create_react_agent, AgentExecutor

# Load Environment Variables
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHER_STACK_API_KEY = os.getenv("WEATHER_STACK_API_KEY")

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(
    page_title="",
    page_icon="🌍🌤️🤖",
    layout="centered"
)

st.title("🌍🌤️🤖 AI Web & Weather Assistant")
st.write("Ask anything about capitals and weather.")

# ----------------------------
# Search Tool
# ----------------------------
search_tool = TavilySearchResults(max_results=2)

# ----------------------------
# Weather Tool
# ----------------------------
@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """
    try:
        url = (
            f"https://api.weatherstack.com/current?"
            f"access_key={WEATHER_STACK_API_KEY}&query={city}"
        )

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        if "current" not in data:
            return f"Could not fetch weather data for {city}"

        return (
            f"City: {city}\n"
            f"Temperature: {data['current']['temperature']}°C\n"
            f"Weather: {data['current']['weather_descriptions'][0]}\n"
            f"Humidity: {data['current']['humidity']}%"
        )

    except requests.exceptions.RequestException as e:
        return f"Weather API Error: {str(e)}"

# ----------------------------
# LLM
# ----------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0
)

# ----------------------------
# Prompt
# ----------------------------
prompt = hub.pull("hwchase17/react")

# ----------------------------
# Tools
# ----------------------------
tools = [search_tool, get_weather_data]

# ----------------------------
# Agent
# ----------------------------
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# ----------------------------
# Agent Executor
# ----------------------------
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# ----------------------------
# User Input
# ----------------------------
user_input = st.text_input(
    "Enter your question:",
    placeholder="Example: Find the capital of Pakistan and its current weather"
)

# ----------------------------
# Run Agent
# ----------------------------
if st.button("Get Answer"):

    if user_input.strip() == "":
        st.warning("Please enter a question.")

    else:
        with st.spinner("Thinking..."):
            try:
                response = agent_executor.invoke({
                    "input": user_input
                })

                st.success("Answer Generated")
                st.write(response["output"])

            except Exception as e:
                st.error(f"Error: {str(e)}")