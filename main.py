import streamlit as st
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
import os
load_dotenv()
os.environ['google_api_key'] = os.getenv('PALM_API_KEY')
llm = GooglePalm()
llm.temperature = 0.1
# Title
st.title(" Welcome to  our restaurant")

# Selectbox
selected_cuisine = st.selectbox("Select a cuisine:", ['Italian', 'French', 'Japanese', 'Indian', 'Mexican'])

# Display selected cuisine
st.write(f"restaurant, names for  {selected_cuisine} cuisine.")

prompt = PromptTemplate.from_template("Suggest restronant name for  {product}?")
chain1 = LLMChain(llm =llm,prompt = prompt)
rest_name= chain1.run(selected_cuisine)
st.title(rest_name)

prompt = PromptTemplate.from_template("Food items in that restorant  {product}?")
chain2 = LLMChain(llm =llm,prompt = prompt)
st.write("Our Menu :")
st.write(chain2.run(rest_name))

prompt = PromptTemplate.from_template("suggest price in indian ruppee  {product}?")
chain3= LLMChain(llm =llm,prompt = prompt)

Chain  = SimpleSequentialChain(chains = [chain1,chain2,chain3],verbose=True)
st.write(Chain.run(selected_cuisine))


