# Import required libraries
# we store sensitive info like API keys in a .env file.
# this keeps secrets out of the code(best practice)
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from the .env file
# Get the directory where this Python file is located
script_dir = Path(__file__).resolve().parent

# Create the path to the .env file
env_path = script_dir / ".env"

# Load variables from the .env file into the environment
load_dotenv(dotenv_path=env_path)

# -------------------------------------------------------
# Verify that the API key was loaded successfully
# -------------------------------------------------------

# Fetch the API key from the environment
api_key = os.getenv("GROQ_API_KEY")

# Check if the key exists
# print("GROQ_API_KEY found:", bool(api_key))

# # Stop the program if the key is missing
# if not api_key:
#     raise ValueError("GROQ_API_KEY not found! Check your .env file.")


# --------------------------
# Step 1 : MODEL (The Brain)
# --------------------------
from langchain_groq import ChatGroq
llm = ChatGroq(
    # LLM model to use
    model="llama-3.3-70b-versatile",

    # API key used for authentication
    api_key=api_key,

    # Controls randomness
    # 0 = deterministic (same answer every time)
    # Higher values = more creative responses
    temperature=0
)

# -----------------------------------------
# Step 2 : Defining the tools ("the hands")
# -----------------------------------------

from langchain_core.tools import tool
import math
@tool
def add(a:float,b:float)->float:
    """
    Add two numbers together.
    The agent will use this when it detects an addition problem.
    """
    return a+b

@tool
def multiply(a:float,b:float)->float:
    """
    Multiply two numbers together.
    The agent will use this when it detects a multiplication task.
    """
    return a*b

@tool
def divide(a:float,b:float)->str:
    """
    Divide the first number with second.
    The agent will raise DivisionByZero error if divided by zero.
    """
    if(b==0):
        return "Error:Cannot divide by zero"
    return a/b

@tool
def square_root(number:float)->str:
    """
    Calculate the square root of a number.
    Includes error handling for negative inputs
    """
    if(number<0):
        return "Error: Cannot calculate square root of a negative number."
    return math.sqrt(number)

# combine all tools into a list
tools=[add,multiply,divide,square_root]

print("=== Available Tools ===")
print()
for t in tools:
    print(f" {t.name}: {t.description}")
print()

# -----------------------------------------
# Step 3 : Create the agent (the "loop")
# -----------------------------------------
# create_agent automatically builds the ReAct loop : 

# 1.Reason->LLM decides what to do 
# 2.Act->calls a tool (if needed)
# 3.observe->gets the result
# 4.Repeat until done
# you dont have to manually write the loop - the framework does it

from langchain.agents import create_agent

agent=create_agent(
    model=llm,
    tools=tools,
)

# ------------------------
# Step 4 : Run the agent
# ------------------------

def run_agent(question : str):
    """Run the agent and print a clean,beginner-friendly execution trace."""

    print(f"\n😊 User : {question}")
    print("-" * 60)

    result=agent.invoke({
        "messages":[("user",question)]
    })

    print("🔎 Clean Agent Execution Trace")
    print("-" * 60)

    step=1
    for msg in result["messages"]:
        if msg.type =="human":
            print(f"{step}. User asked : ")
            print(f"  {msg.content}")
            step += 1

        elif msg.type == "ai" and getattr(msg,"tool_calls",None):
            for tool_call in msg.tool_calls:
                tool_name=tool_call["name"]
                tool_args=tool_call["args"]

                print(f"{step}. Agent decision : ")
                print(f"    I need to use the tool :{tool_name} ")
                print(f"    Tool input:{tool_args} ")
                step+=1

        elif msg.type == "tool":
            print(f"{step}. Tool observation : ")
            print(f"    Tool returned : {msg.content}")
            step += 1

        elif msg.type == "ai" and msg.content:
            print(f"{step}. Final answer : ")
            print(f"    {msg.content}")
            step += 1
        
    print("=" * 60)

# TEST CASES :
while True:
    question=input("\nAsk me anything (type 'exit' to quit :)")
    if(question.lower()=="exit"):
        break

    run_agent(question)