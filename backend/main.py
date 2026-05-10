from fastapi import FastAPI
from backend.api.routes import router

app = FastAPI(title="Prof GPT API")

# Connect the routes we just made
app.include_router(router)

@app.get("/")
def home():
    return {"message": "Prof GPT API is running! Go to /docs to test it."}