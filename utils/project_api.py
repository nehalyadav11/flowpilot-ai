import requests
from typing import List, Dict, Any

API_BASE = "http://localhost:3000"

def get_projects() -> List[Dict[str, Any]]:
    try:
        resp = requests.get(f"{API_BASE}/projects")
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []

def create_project(name: str, description: str, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
    payload = {
        "name": name,
        "description": description,
    }
    if start_date:
        payload["startDate"] = start_date
    if end_date:
        payload["endDate"] = end_date

    try:
        resp = requests.post(f"{API_BASE}/projects", json=payload)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}
