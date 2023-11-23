import streamlit as st
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.agents import AgentType,initialize_agent,load_tools
import streamlit as st
import os
os.environ['google_api_key'] = 'AIzaSyBKQUeLi1GlQXUnVewYWEL1k6Vvr3wFE9g'
llm = GooglePalm()
llm.temperature = 0.1

tools = load_tools(['wikipedia'],llm=llm)

st.title("wikiBot")
agent = initialize_agent( tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)

st.write("Ask anything :")
input = st.text_input("Enter your question :")
output = agent.run(input)

st.write(output)