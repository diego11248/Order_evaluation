from typing import List, Optional
from pydantic import BaseModel
import google.generativeai as genai
import os

class ExtraInfo(BaseModel):
    spouse_of_new_president: Optional[str] = None

class GeneratedEdit(BaseModel):
    descriptor: str
    target_new: str
    ground_truth: Optional[str] = None
    extra: Optional[ExtraInfo] = None

def _prompt(seed: dict) -> str:
    return (
        "Given the following fact edit seed, return 5 correlated edits that are entailed.\n"
        "Generalization should update related facts consistently.\n"
        f"Seed: {seed}"
    )

def _configure(api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
    api_key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY env var.")
    genai.configure(api_key=api_key)
    client = genai.Client()
    return client, model

def generate_correlated_edits(seed: dict, api_key: Optional[str] = None) -> List[GeneratedEdit]:
    client, model = _configure(api_key)
    resp = client.models.generate_content(
        model=model,
        contents=[_prompt(seed)],
        config=dict(
            response_mime_type="application/json",
            response_schema=List[GeneratedEdit],
        ),
    )
    return resp.parsed
