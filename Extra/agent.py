# agent.py

from data import learner_profile, labor_market
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Create OpenAI client
client = OpenAI()

# ---------- Skill Gap Analyzer ----------
def analyze_skill_gap(learner, market):
    missing_skills = []

    for skill in market:
        if skill not in learner:
            missing_skills.append(skill)

    return missing_skills


# ---------- Get Market Skills ----------
target_role = learner_profile["goal"]
required_skills = labor_market[target_role]
learner_skills = learner_profile["skills"]

missing_skills = analyze_skill_gap(learner_skills, required_skills)
print("Missing Skills:", missing_skills)


# ---------- Guidance Generator ----------
def generate_guidance(profile, missing_skills):
    prompt = f"""
You are an educational and career guidance AI agent.

Learner Profile:
- Major: {profile['major']}
- Current Skills: {profile['skills']}
- Career Goal: {profile['goal']}
- Level: {profile['level']}

Missing Skills:
{missing_skills}

Generate:
1. An 8-week learning roadmap
2. 3 practical project ideas
3. 2 beginner-friendly research topics
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# ---------- Run Agent ----------
guidance = generate_guidance(learner_profile, missing_skills)

print("\n=== Personalized Guidance ===\n")
print(guidance)
