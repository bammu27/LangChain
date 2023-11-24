import streamlit as st
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.agents import AgentType,initialize_agent,load_tools
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()
os.environ['google_api_key'] = os.getenv('PALM_API_KEY')
print(os.getenv('PALM_API_KEY'))
llm = GooglePalm()
llm.temperature = 0.1

tools = load_tools(['wikipedia'],llm=llm)

st.title("wikiBot")
agent = initialize_agent( tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)

st.write("Ask anything :")
input = st.text_input("Enter your question :")
st.write(input)
output = agent.run(input)

st.write(output)