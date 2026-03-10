import os
from dotenv import load_dotenv
import openai
import pandas as pd

load_dotenv()
api_key = os.getenv("API_KEY")
print(api_key)
client = openai.OpenAI(api_key=api_key)

software_requirements = pd.read_csv("cleaned_requirements.csv") # replace cleaned_requirements.csv with actual csv file of the clean dataset

response = client.chat.completions.create( #this is chain1. 
    model = "gpt-5",
    messages = [
            {"role": "user", "content": "prompt1 goes here...{software_requirements}"} #prompt for chian1. 
    ]
)
chain1 = response.choices[0].message.content
print(chain1)

response2 = client.chat.completions.create( #chain2
     model = "gpt-5",
    messages = [
            {"role": "user", "content": "Prompt2 goes here.... {chain1}"} #prompt2
    ]
)
chain2 = response2.choices[0].message.content
print(chain2)


response3 = client.chat.completions.create( #chain3
     model = "gpt-5",
    messages = [
            {"role": "user", "content": "Prompt3 goes here.... {chain2}"} #prompt2
    ]
)
chain3 = response3.choices[0].message.content
print(chain3)