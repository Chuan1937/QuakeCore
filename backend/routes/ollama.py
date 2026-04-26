"""Ollama model detection route."""

import httpx
from fastapi import APIRouter

router = APIRouter(prefix="/api/ollama", tags=["ollama"])


@router.get("/models")
async def list_ollama_models(base_url: str = "http://localhost:11434"):
    url = base_url.rstrip("/") + "/api/tags"

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            models = [
                item.get("name")
                for item in data.get("models", [])
                if item.get("name")
            ]
            return {
                "ok": True,
                "models": models,
                "message": "检测成功" if models else "未检测到本地 Ollama 模型",
            }
    except httpx.ConnectError:
        return {
            "ok": False,
            "models": [],
            "message": "未检测到 Ollama 服务，请确认 Ollama 已启动",
        }
    except httpx.TimeoutException:
        return {
            "ok": False,
            "models": [],
            "message": "连接 Ollama 超时，请确认服务地址是否正确",
        }
    except Exception as e:
        return {
            "ok": False,
            "models": [],
            "message": f"Ollama 检测失败：{e}",
        }
