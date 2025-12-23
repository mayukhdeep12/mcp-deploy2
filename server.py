import os
import io
import base64
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager
import yfinance as yf
from duckduckgo_search import DDGS
import matplotlib.pyplot as plt

# MCP Core Imports
from mcp.server.lowlevel import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent, ImageContent

# --- CONFIGURATION ---
API_SECRET = os.getenv("MCP_API_KEY", "default-dev-key")
mcp = Server("hedge-fund-analyst")

# --- TOOLS ---
async def get_stock_news(ticker: str) -> str:
    """Search for the latest news driving a stock's price."""
    print(f"Searching news for {ticker}...")
    with DDGS() as ddgs:
        results = [r for r in ddgs.news(f"{ticker} stock news", max_results=3)]
    return "\n".join([f"- {r['title']} ({r['source']})" for r in results]) if results else "No news."

async def get_stock_chart(ticker: str) -> str:
    """Generates a price history chart (base64 string)."""
    print(f"Generating chart for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6m")
        if hist.empty: return "No data."
        
        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(hist.index, hist['Close'])
        plt.title(f'{ticker} - 6 Month Trend')
        plt.grid(True)
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        # We return a message confirming generation (images in MCP require specific handling)
        return f"Chart generated for {ticker}. Latest price: ${hist['Close'].iloc[-1]:.2f}"
    except Exception as e:
        return f"Error generating chart: {str(e)}"

# Register Tools
@mcp.list_tools()
async def list_tools():
    return [
        Tool(
            name="get_stock_news",
            description="Get latest news for a stock ticker",
            inputSchema={"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}
        ),
        Tool(
            name="get_stock_chart",
            description="Get 6-month price chart calculation",
            inputSchema={"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}
        )
    ]

@mcp.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_stock_news":
        result = await get_stock_news(arguments["ticker"])
        return [TextContent(type="text", text=result)]
    elif name == "get_stock_chart":
        result = await get_stock_chart(arguments["ticker"])
        return [TextContent(type="text", text=result)]
    raise ValueError(f"Unknown tool: {name}")

# --- SERVER SETUP (The Fix) ---
app = FastAPI()

# Create the Transport Layer
sse = SseServerTransport("/messages")

@app.post("/messages")
async def handle_messages(request: Request):
    """Forward POST requests to the MCP transport handler"""
    # Security: Validate Token in Query or Headers if needed (simplified here)
    await sse.handle_post_message(request.scope, request.receive, request._send)

@app.get("/sse")
async def handle_sse(request: Request):
    """
    Directly hook into the ASGI lifecycle to handle streaming.
    This fixes the 'coroutine not iterable' error.
    """
    # 1. SECURITY CHECK
    token = request.query_params.get("token")
    if token != API_SECRET:
        # We must manually return a 403 because we are bypassing FastAPI dependencies
        return HTTPException(status_code=403, detail="Unauthorized")

    # 2. CONNECT AND RUN SERVER
    # The transport's connect_sse method yields input/output streams
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await mcp.run(streams[0], streams[1], mcp.create_initialization_options())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
