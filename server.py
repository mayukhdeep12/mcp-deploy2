import io
import base64
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, Depends
from fastmcp import FastMCP
import yfinance as yf
from duckduckgo_search import DDGS
import matplotlib.pyplot as plt

# --- 1. SECURITY CONFIGURATION ---
# We read the secret key from the environment variables (set in Render later)
API_SECRET = os.getenv("MCP_API_KEY", "default-dev-key")

def verify_token(request: Request):
    """
    Middleware to check if the URL has ?token=MY_SECRET_KEY
    """
    token = request.query_params.get("token")
    if token != API_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized: Invalid API Key")
    return token

# --- 2. FASTMCP SETUP ---
# We create the MCP server but don't run it yet.
mcp = FastMCP("Hedge Fund Analyst")

# --- 3. TOOLS ---

@mcp.tool()
def get_stock_news(ticker: str) -> str:
    """Search for the latest news driving a stock's price."""
    print(f"Searching news for {ticker}...")
    with DDGS() as ddgs:
        results = [r for r in ddgs.news(f"{ticker} stock news", max_results=3)]
    
    if not results:
        return "No news found."
    
    formatted = "\n".join([f"- {r['title']} ({r['source']})" for r in results])
    return f"Latest News for {ticker}:\n{formatted}"

@mcp.tool()
def get_stock_chart(ticker: str) -> str:
    """
    Generates a price history chart for a stock over the last 6 months.
    Returns a base64 string of the image.
    """
    print(f"Generating chart for {ticker}...")
    
    # 1. Fetch Data
    stock = yf.Ticker(ticker)
    hist = stock.history(period="6m")
    
    if hist.empty:
        return "Error: No data found for ticker."

    # 2. Create Plot (Matplotlib)
    plt.figure(figsize=(10, 5))
    plt.plot(hist.index, hist['Close'], label=f'{ticker} Close Price')
    plt.title(f'{ticker} - 6 Month Trend')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.grid(True)
    plt.legend()
    
    # 3. Save to Memory Buffer (not a file)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    # 4. Convert to Base64
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    # FastMCP automatically detects if you return an Image object, 
    # but for simplicity in this advanced tutorial, we return a special 
    # string format that Claude knows how to interpret if we tell it 
    # or we can return the raw image via FastMCP's context capabilities.
    # To keep it simple: We return a message saying the chart is ready.
    # Note: Currently, displaying inline images in Claude via MCP is 
    # dependent on the client support. We will return text data for now
    # or a "Data URL".
    
    return f"Chart generated. (In a full GUI client, this would render). Price moved from {hist['Close'].iloc[0]:.2f} to {hist['Close'].iloc[-1]:.2f}"

# --- 4. ADVANCED: MOUNTING ON FASTAPI ---
# This is how we add Authentication to FastMCP.
# We create a standard FastAPI app and "mount" the MCP server on it.

app = FastAPI()

# We expose the MCP server on /sse and /messages, PROTECTED by our verify_token
mcp.mount_as_sse(app, path="/sse", dependencies=[Depends(verify_token)])
mcp.mount_as_messages(app, path="/messages", dependencies=[Depends(verify_token)])

@app.get("/health")
def health():
    return {"status": "healthy"}