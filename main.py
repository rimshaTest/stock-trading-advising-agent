from datetime import datetime
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_classic.output_parsers import OutputFixingParser

load_dotenv()  # Load environment variables from .env file

class StockRecommendation(BaseModel):
    ticker: str
    action: Literal["buy", "sell", "hold"]
    reason: str

class ResponseModel(BaseModel):
    stock: list[StockRecommendation] = Field(description="List of stock recommendations with tickers, actions, and reasons")
    sources: list[str]
    tools_used: list[str]

llm = ChatOllama(model="llama3.1:latest")  # Initialize the Ollama model
parser = PydanticOutputParser(pydantic_object=ResponseModel)
parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You're a stock trading advising agent. You will be given a list of the current stocks held by the user. Your task is to use necessary tools to research stock trends as of date ({today}), give lists of sources and tools used for the recommendations, and suggest actions for each of the stocks held and the reasons behind them. Wrap the output strictly in this format and provide no other text: \n{parser_format_instructions}"),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}")

]).partial(
    today = datetime.now().strftime("%Y-%m-%d"),
    parser_format_instructions = parser.get_format_instructions()
)


agent = create_tool_calling_agent(llm, tools=[], prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)
raw_response = agent_executor.invoke({"query": "currentStocks: AAPL, GOOG, MSFT"})

structured_response = parser.parse(raw_response.get("output", ""))
print(structured_response)