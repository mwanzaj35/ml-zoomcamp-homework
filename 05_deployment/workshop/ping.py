from fastapi import FastAPI
import uvicorn

# define the application
app = FastAPI(title="ping")

# /ping is url used to access application
@app.get("/ping")
def ping():
    return "PONG"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)