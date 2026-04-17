from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import List, Dict
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent

from collections import Counter
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from main import LearnerIntelligenceAgent
from main2 import LaborIntelligenceAgent

load_dotenv()

# ✅ CREATE APP FIRST
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# Root Health Check
# -------------------------
@app.get("/")
def root():
    return {"status": "Backend is running"}

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/ui", response_class=HTMLResponse)
def serve_ui():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

# -------------------------
# Job Data
# -------------------------
JOB_POSTINGS = [
    {
        "title": "Data Scientist",
        "description": "Experience with Python, SQL, Machine Learning, and statistics",
        "date": "2026-01-15"
    },
    {
        "title": "AI Engineer",
        "description": "Strong Python skills, deep learning, NLP, and cloud experience",
        "date": "2026-01-20"
    },
    {
        "title": "Data Analyst",
        "description": "Excel, SQL, Power BI, data visualization",
        "date": "2026-01-10"
    }
]

# -------------------------
# Tools
# -------------------------
@tool
def extract_skills(job_description: str):
    """
    Extract technical and professional skills from a job description.
    Returns a list of skill names.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0.4)

    prompt = f"""
    Extract a list of technical and professional skills.
    Return only a Python list.
    The output must be in egyptian Arabic even the input is in any language.

    - القيم لازم تكون بالمصري الواضح (صياغة طبيعية، مش تهجئة بالحروف)، من غير ما تضيع المعنى.
    - استخدم نص فاضي لأي قيمة مش مذكورة.
    - استخدم مصفوفة فاضية للمهارات لو مش مذكورة.
    - وحّد العناصر وشيل التكرار.
    - ما تخترعش بيانات (شركة/مكان/تاريخ) لو مش موجودة.
    - لو المدخل قائمة وظائف، استخرج كل وظيفة كعنصر مستقل.
    - لو المدخل وصف عام واحد، اعمله كوظيفة واحدة بعنوان "غير محدد" مع وصف مناسب.
    - لو فيه تواريخ بصيغة واضحة، سجّلها كما هي.
    - استخرج المهارات الضمنية من الوصف (مثال: "اشتغلت SQL" → "SQL").
    - خلي عناصر المهارات عبارات قصيرة (1–4 كلمات) وبالمصري الواضح.

    انت ذكاء اصطناعي لسوق العمل ولترشيح كوسات ومساعده الجميع في التعلم


    Job description:
    {job_description}
    """

    response = llm.invoke(prompt)
    try:
        return json.loads(response.content)
    except:
        return []

@tool
def calculate_demand(job_titles: List[str]) -> Dict[str, int]:
    """
    Calculate demand score (0–100).
    """
    counts = Counter(job_titles)
    max_count = max(counts.values())
    return {k: int((v / max_count) * 100) for k, v in counts.items()}

@tool
def analyze_labor_market(jobs: List[dict]) -> dict:
    """
    Analyze labor market demand and skills.
    """
    role_skills = {}
    titles = []

    for job in jobs:
        titles.append(job["title"])
        skills = extract_skills.run(job["description"])
        role_skills.setdefault(job["title"], []).extend(skills)

    demand = calculate_demand.run(titles)

    return {
        role: {
            "top_skills": Counter(skills).most_common(5),
            "demand_score": demand.get(role, 0)
        }
        for role, skills in role_skills.items()
    }

# -------------------------
# Agent
# -------------------------
TOOLS = [extract_skills, calculate_demand, analyze_labor_market]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent = create_agent(
    llm,
    TOOLS,
    system_prompt="You are a Labor Market Intelligence Agent, The output must be in egyptian Arabic even the input is in any language."
)

# Initialize the LearnerIntelligenceAgent
learner_agent = LearnerIntelligenceAgent()

# Initialize the LaborIntelligenceAgent
labor_agent = LaborIntelligenceAgent()

# -------------------------
# API Schema
# -------------------------
class UserQuery(BaseModel):
    message: str

# -------------------------
# API Endpoints
# -------------------------
@app.post("/run-agent")
def run_agent_api(payload: dict = Body(...)):
    user_message = payload.get("message", "")

    response = agent.invoke({
        "messages": [HumanMessage(content=user_message)]
    })

    return {"response": response["messages"][-1].content}

@app.post("/analyze")
def analyze(query: UserQuery):
    response = agent.invoke({
        "messages": [HumanMessage(content=query.message)]
    })
    return {"response": response["messages"][-1].content}

@app.post("/process-input")
def process_input(payload: dict = Body(...)):
    """
    Endpoint to process user input using LearnerIntelligenceAgent.
    """
    user_input = payload.get("message", "")
    if not user_input:
        return {"error": "No input provided."}

    try:
        # Process the input using the agent
        output = learner_agent.infer(user_input)
        return {"response": output}
    except Exception as e:
        return {"error": str(e)}

@app.post("/process-learner")
def process_learner(payload: dict = Body(...)):
    """
    Endpoint to process user input using LearnerIntelligenceAgent.
    """
    user_input = payload.get("message", "")
    if not user_input:
        return {"error": "No input provided."}

    try:
        # Process the input using the LearnerIntelligenceAgent
        output = learner_agent.infer(user_input)
        return {"response": output}
    except Exception as e:
        return {"error": str(e)}

@app.post("/process-market")
def process_market(payload: dict = Body(...)):
    """
    Endpoint to process user input using LaborIntelligenceAgent.
    """
    user_input = payload.get("message", "")
    if not user_input:
        return {"error": "No input provided."}

    try:
        # Process the input using the LaborIntelligenceAgent
        output = labor_agent.infer(user_input)
        return {"response": output}
    except Exception as e:
        return {"error": str(e)}
