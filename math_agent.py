from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
import math

# Loading environment variables
load_dotenv()

@tool
def square_root(x: float) -> float:
    """Calcuate the square root of a number."""
    return x ** 0.5

@tool
def square(x: float) -> float:
    """Calculate the square of a number."""
    return x ** 2

@tool
def add_numbers(x: float, y: float) -> float:
    """Add two numbers together and return the result."""
    return x + y

@tool
def subtract_numbers(x: float, y: float) -> float:
    """Subtract two numbers and return the result."""
    return x - y

@tool
def multiply_number(x: float, y: float) -> float:
    """Multiply two numbers together and return the result."""
    return x * y

@tool
def divide_numbers(x: float, y: float) -> float:
    """Divide two numbers and return the result."""
    return x / y

@tool
def factorial(x: float) -> float:
    """Calculate the factorial of a number."""
    # Calculate factorial using recursion
    if x < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    elif x == 0 or x == 1:
        return 1
    else:
        return x * factorial(x - 1)

model = init_chat_model(
    model="gpt-4o-mini",
    max_tokens=1024,
    temperature=0.5
)

# Creating subagents for the "main agent"
subagent_1 = create_agent(
    model=model,
    tools=[square_root, square]
)

subagent_2 = create_agent(
    model=model,
    tools=[add_numbers, subtract_numbers]
)

subagent_3 = create_agent(
    model=model,
    tools=[multiply_number, divide_numbers]
)

subagent_4 = create_agent(
    model=model,
    tools=[factorial]
)

# Creating my subagents as tools for my main agent to access - ensure that we use query: str -> str to match the expected format since we are taking a user input
@tool
def subagent_1_tool(query: str) -> str:
    """Subagent 1 tool for calculating square roots and squares of a number"""
    result = subagent_1.invoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content

@tool
def subagent_2_tool(query: str) -> str:
    """Subagent 2 tool for adding and subtracting numbers."""
    result = subagent_2.invoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content

@tool 
def subagent_3_tool(query: str) -> str:
    """Subagent 3 tool for multiplying and dividing numbers."""
    result = subagent_3.invoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content

@tool
def subagent_4_tool(query: str) -> str:
    """Subagent 4 tool for calculating factorials of numbers."""
    result = subagent_4.invoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content

system_prompt = """

You are a helpful assistant that specializes in mathematics and science. You are designed to help users with their queries and provide accurate and helpful responses.
You are equipped with a variety of tools to assist with your tasks.

Please keep the structure of your responses consistent with the following format:

Question: [User's question]

Answer: [Your answer]

Explanation: [Explanation of your answer and how you arrived at the solution, what was the math involved.]

"""

# Creating the "main agent"
agent = create_agent(
    model=model,
    tools=[subagent_1_tool, subagent_2_tool, subagent_3_tool],
    system_prompt=system_prompt,
    checkpointer=InMemorySaver()
)

# "config" is used to hold memory and other stateful information
config = {"configurable": {"thread_id": "1"}}
while True:
    user_question = input("User: ")

    if user_question.lower() in ["exit", "quit"]:
        break

    response = agent.invoke(
        {"messages": [HumanMessage(content=user_question)]},
        config,
    )

    # Hard response - looks kinda weird when you sit for 20 seconds and dont get a response but it works the best :)
    print(f"Assistant\n{response['messages'][-1].content}\n")


    '''
    # Streaming the response - a little bit nicer/interactive way of getting the response **LOOKS REALLY WEIRD**
    for token, metadata in agent.stream (
        {"messages": [HumanMessage(content="What is the square root of 125?")]},
        stream_mode="messages"
    ):

        print(f"{token.content}", end="")
        #print(f"\n{metadata}\n")

    '''
