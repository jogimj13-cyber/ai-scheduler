# fastapi dev .\02_stream_server.py
import sys
import os

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
print(root_dir)

from dotenv import load_dotenv

load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import json

from langchain_openrouter import ChatOpenRouter
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage, AIMessageChunk
from utils import load_mcp_config

checkpointer = InMemorySaver()
tools = None

# Pydantic Data Model
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=2)
    model: str = "gemini-2.5-flash"
    thread_id: str = "default"


async def get_tools():
    mcp_config = load_mcp_config("google-calendar")
    # print("mcp config loaded:", mcp_config)

    client = MultiServerMCPClient(mcp_config)
    mcp_tools = await client.get_tools()

    # # Filter tools that work with Gemini

    print(f"Loaded {len(mcp_tools)} Tools")
    print(f"Tools Available\n{[tool.name for tool in mcp_tools]}")

    return mcp_tools


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tools
    tools = await get_tools()
    print("Tools are loaded. ready to create agent!")
    yield


app = FastAPI(lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def stream_response(query, model_name, thread_id):
    system_prompt = """
You are an AI agent that is responsible for scheduling tasks and events using Google Calendar.

Use the following tools to schedule tasks and events:
 - list-calendars: List all available calendars
 - list-events: List events with date filtering
 - get-event: Get details of a specific event by ID
 - search-events: Search events by text query
 - create-event: Create new calendar events
 - update-event: Update existing events
 - delete-event: Delete events
 - respond-to-event: Respond to event invitations (Accept, Decline, Maybe, No Response)
 - get-freebusy: Check availability across calendars, including external calendars
 - get-current-time: Get current date and time in calendar's timezone
 - list-colors: List available event colors
 - manage-accounts: Add, list, or remove connected Google accounts

Guidelines:
 - Add and read freely, but never delete without confirmation by the user.
 - Always use the most appropriate tool for the task.
 - If the user does not have a connected Google Calendar account, use the manage-accounts tool to add one.
 - Use the format below: 
    manage-accounts with action: "add", account_id: given_id
    manage-accounts with action: "list"
    manage-accounts with action: "delete", account_id: given_id
 - Once the account is added, give the user the URL returned by the manage-accounts tool to connect to the Google Calendar account.
 - Always respond with efficient and effective responses.
 - Always respond with a summary of the task and event scheduling.
 - Always respond in the fastest possible time.
 - Always respond in Markdown format.
"""
    # Initialize model and agent
    model = ChatOpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"), model=model_name)
    agent = create_agent(model=model, tools=tools, system_prompt=system_prompt, checkpointer=checkpointer)

    # Configuration with thread ID for conversation memory
    config = {"configurable": {"thread_id": thread_id}}

    async for chunk, metadata in agent.astream(
        {'messages':[HumanMessage(query)]},
        stream_mode='messages', config=config):

        data = {
            "type": chunk.__class__.__name__,
            "content": chunk.text
        }

        if isinstance(chunk, AIMessageChunk) and chunk.tool_calls:
            data['tool_calls'] = chunk.tool_calls

        # send json response
        yield (json.dumps(data) + "\n").encode()


@app.get("/")
async def read_root():
    return {"Hello": "MJ. Your FastAPI Server is up!"}


@app.post("/chat_stream")
async def chat_stream(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Empty prompt!")
    
    try:
        return StreamingResponse(
            stream_response(request.query, request.model, request.thread_id),
            media_type="application/x-ndjson")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app=app, host="0.0.0.0", port=8002)
