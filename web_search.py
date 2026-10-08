import os

from langchain_community.utilities import SerpAPIWrapper

from config import SERPAPI_API_KEY


def search_web(query: str) -> str:
    if not SERPAPI_API_KEY:
        raise RuntimeError("SERPAPI_API_KEY is not configured.")

    search = SerpAPIWrapper(
        serpapi_api_key=SERPAPI_API_KEY,
    )

    results = search.run(query)

    return results