import os
import json
import io
import base64
import asyncio

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import Response
from sse_starlette.sse import EventSourceResponse

# We use the OFFICIAL Core SDK now (Advanced Level)
from mcp.server.lowlevel import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
import yfinance as yf
from duckduckgo_search import DDGS
import matplotlib.pyplot as plt

# --- 1. SETUP SERVER & APP ---
app = FastAPI()
mcp_server = Server("hedge-fund-analyst")

# Security Token
API_SECRET = os.getenv("MCP_API_KEY", "default-dev-key")

# --- 2. DEFINE TOOLS (The "Pro" Way) ---
# In the core SDK, we register tools manually or use decorators if available.
# We will define the logic and then register them.

async def get_stock_news(ticker: str) -> str:
    """Search for the latest news driving a stock's price."""
    print(f"Searching news for {ticker}...")
    with DDGS() as ddgs:
        results = [r for r in ddgs.news(f"{ticker} stock news", max_results=3)]
    
    if not results:
        return "No news found."
    return "\n".join([f"- {r['title']} ({r['source']})" for r in results])

async def get_stock_chart(ticker: str) -> str:
    """Generates a price history chart for a stock (base64)."""
    print(f"Generating chart for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6m")
        if hist.empty: return "No data."
        
        plt.figure(figsize=(10, 5))
        plt.plot(hist.index, hist['Close'])
        plt.title(f'{ticker} - 6 Month Trend')
        plt.grid(True)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return f"Chart generated successfully. Closing price: {hist['Close'].iloc[-1]:.2f}"
    except Exception as e:
        return f"Error: {str(e)}"

# Register the tools with the MCP Server
@mcp_server.list_tools()
async def list_tools():
    return [
        Tool(
            name="get_stock_news",
            description="Get latest news for a stock ticker",
            inputSchema={
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"]
            }
        ),
        Tool(
            name="get_stock_chart",
            description="Get 6-month price chart calculation",
            inputSchema={
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"]
            }
        )
    ]

@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_stock_news":
        result = await get_stock_news(arguments["ticker"])
        return [TextContent(type="text", text=result)]
    
    elif name == "get_stock_chart":
        result = await get_stock_chart(arguments["ticker"])
        return [TextContent(type="text", text=result)]
    
    raise ValueError(f"Unknown tool: {name}")

# --- 3. AUTHENTICATION ---
async def verify_token(request: Request):
    token = request.query_params.get("token")
    if token != API_SECRET:
        raise HTTPException(status_code=403, detail="Invalid API Token")
    return token

# --- 4. THE SSE ENDPOINTS (The Bridge) ---
# This dictionary holds active connections
sse_transports = {}

@app.get("/sse")
async def handle_sse(request: Request, token: str = Depends(verify_token)):
    """
    This is the endpoint Claude connects to.
    """
    async def event_generator():
        # Create a transport for this specific connection
        transport = SseServerTransport("/messages")
        
        async with transport.connect_sse(request.scope, request.receive, request._send) as streams:
            # Store transport to handle incoming POST messages later
            sse_transports[token] = transport
            
            # Run the server for this connection
            await mcp_server.run(streams[0], streams[1], mcp_server.create_initialization_options())
            
    return EventSourceResponse(event_generator())

@app.post("/messages")
async def handle_messages(request: Request):
    """
    Claude sends responses back to us here. 
    Note: In a full implementation, we need to map sessions better.
    For this single-user demo, we assume the active transport.
    """
    # In a real app, you'd extract the session ID from the URL. 
    # For this demo, we pick the most recent transport (Simplification)
    if not sse_transports:
        raise HTTPException(404, "No active session")
        
    # Pick the first available transport (works for single user demo)
    transport = list(sse_transports.values())[0]
    
    await transport.handle_post_message(request.scope, request.receive, request._send)
    return Response(status_code=202)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
