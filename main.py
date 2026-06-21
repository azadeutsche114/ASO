from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional
import app.model as model_module
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_module.startup()
    yield

app = FastAPI(title="OligoAI ASO Predictor", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


class PredictRequest(BaseModel):
    target_input: str
    chemistry:    Optional[str]   = "5_10_5_MOE"
    dosage_nm:    Optional[float] = 4000.0
    transfection: Optional[str]   = "lipofection"

class PredictResponse(BaseModel):
    description: str
    candidates:  list

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        candidates, desc = model_module.run_prediction(
            target_input=req.target_input,
            chemistry=req.chemistry,
            dosage_nm=req.dosage_nm,
            transfection=req.transfection,
        )
        return PredictResponse(description=desc, candidates=candidates)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))