"""
Agent ecommerce - AI Tinkerers Hackathon "Agents, Everywhere"
Stack : Python + Streamlit + OpenAI SDK (function calling) + SerpApi (Google Shopping)

Installation :
    pip install streamlit openai requests

Lancement local :
    streamlit run app.py

Variables d'environnement à définir (fichier .env ou secrets Streamlit) :
    OPENAI_API_KEY
    SERPAPI_KEY

Déploiement :
    Streamlit Community Cloud (share.streamlit.io) est le plus simple et gratuit
    pour ce type d'app. Vercel ne supporte pas Streamlit nativement.
    Alternatives : Hugging Face Spaces, Railway, Render.
"""

import json
import os

import requests
import streamlit as st
from openai import OpenAI

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Assistant Achat IA", page_icon="🛒")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
SERPAPI_KEY = os.getenv("SERPAPI_KEY") or st.secrets.get("SERPAPI_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------------------------------------
# 1. PROMPT SYSTÈME
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Tu es un assistant d'achat intégré à une boutique en ligne.
Ton rôle : aider l'utilisateur à trouver le meilleur produit selon ses besoins
(budget, caractéristiques, marque) en utilisant la recherche produit en temps réel.

Règles :
- Utilise TOUJOURS l'outil "search_products" avant de recommander un produit précis
  ou de donner un prix. Ne jamais inventer de prix ou de disponibilité.
- Si l'utilisateur ne précise pas de budget, demande-le avant de lancer une recherche
  large (pour éviter des résultats non pertinents).
- Présente 2 à 4 options maximum, avec pour chacune : nom, prix, et une raison
  courte de la recommander.
- Réponds toujours dans la langue de l'utilisateur.
- Reste concis : c'est un chat, pas une fiche produit.
- Si aucun résultat ne correspond au budget, propose d'élargir la recherche plutôt
  que d'inventer une alternative."""

# ---------------------------------------------------------------------------
# 2. SCHEMA DE FUNCTION CALLING POUR SERPAPI
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": (
                "Recherche des produits en temps réel via Google Shopping (SerpApi). "
                "À utiliser dès que l'utilisateur demande un produit, une comparaison "
                "de prix, ou une recommandation d'achat."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Terme de recherche produit, ex: 'casque bluetooth sport'",
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Prix maximum en euros/devise locale (optionnel)",
                    },
                    "min_price": {
                        "type": "number",
                        "description": "Prix minimum (optionnel)",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Nombre de résultats à retourner (défaut: 5, max: 10)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    }
]


# ---------------------------------------------------------------------------
# 3. APPEL SERPAPI (Google Shopping)
# ---------------------------------------------------------------------------
def search_products(query, max_price=None, min_price=None, num_results=5):
    params = {
        "engine": "google_shopping",
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": min(num_results, 10),
    }

    try:
        res = requests.get("https://serpapi.com/search.json", params=params, timeout=15)
        res.raise_for_status()
    except requests.RequestException as e:
        return {"error": f"SerpApi a échoué : {e}"}

    data = res.json()
    results = []
    for r in data.get("shopping_results", []):
        price = r.get("extracted_price") or r.get("price")
        results.append(
            {
                "title": r.get("title"),
                "price": price,
                "source": r.get("source"),
                "link": r.get("link"),
                "rating": r.get("rating"),
            }
        )

    if max_price:
        results = [r for r in results if not r["price"] or r["price"] <= max_price]
    if min_price:
        results = [r for r in results if not r["price"] or r["price"] >= min_price]

    return {"results": results[:num_results]}


FUNCTION_REGISTRY = {"search_products": search_products}


# ---------------------------------------------------------------------------
# 4. BOUCLE DE FUNCTION CALLING
# ---------------------------------------------------------------------------
def run_agent(messages):
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=conversation,
        tools=TOOLS,
        tool_choice="auto",
    )
    message = response.choices[0].message

    while message.tool_calls:
        # Convertit l'objet message en dict pour l'ajouter à l'historique envoyé à l'API
        conversation.append(message.model_dump(exclude_unset=True))

        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            fn = FUNCTION_REGISTRY.get(fn_name)
            result = fn(**args) if fn else {"error": "Outil inconnu"}

            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=conversation,
            tools=TOOLS,
            tool_choice="auto",
        )
        message = response.choices[0].message

    return message.content


# ---------------------------------------------------------------------------
# 5. INTERFACE STREAMLIT
# ---------------------------------------------------------------------------
st.title("🛒 Assistant Achat IA")
st.caption("Décris ce que tu cherches, je compare les prix en temps réel.")

if not OPENAI_API_KEY or not SERPAPI_KEY:
    st.warning("Configure OPENAI_API_KEY et SERPAPI_KEY dans les secrets Streamlit.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ex: un casque bluetooth sport à moins de 50€"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours..."):
            reply = run_agent(st.session_state.messages)
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})