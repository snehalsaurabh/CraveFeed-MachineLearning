from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import random
import os

# FastAPI app
app = FastAPI()

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model to validate input data
class UserInput(BaseModel):
    id: str
    type: str
    dish: str
    city: str
    followerCount: int

class InputData(BaseModel):
    currentUser: UserInput
    otherUsers: list[UserInput]

# Function to calculate distance between two cities using GPT
openai_api_key=os.getenv("OPENAI_API_KEY")

def gpt_city_distance(city1, city2, openai_api_key):
    try:
        prompt_template = PromptTemplate(
            input_variables=["city1", "city2"],
            template="Estimate the distance in kilometers between {city1} and {city2} in India."
        )
        prompt = prompt_template.format(city1=city1, city2=city2)

        chat_model = ChatOpenAI(
            openai_api_key=openai_api_key,
            model="gpt-3.5-turbo",
            temperature=0,
            max_retries=3
        )
        response = chat_model([HumanMessage(content=prompt)])
        distance_str = response.content.strip()

        try:
            distance = float(distance_str.split()[0])
            return distance
        except ValueError:
            return 1000  # Default value

    except Exception as e:
        return 1000  # Default value in case of any error

# Function to calculate similarity score
def calculate_similarity(current_user: UserInput, other_user: UserInput, openai_api_key: str):
    similarity_score = 0

    # Dish similarity (High priority)
    if current_user.dish.lower() == other_user.dish.lower():
        similarity_score += 50
    else:
        # Add minor penalty for different dishes
        similarity_score -= 5  

    # Location similarity using GPT for estimating distance
    location_distance = gpt_city_distance(current_user.city, other_user.city, openai_api_key)
    location_similarity = 1 / (1 + location_distance)
    similarity_score += location_similarity * 10  # Small weight for location

    # Followers count influence (with logarithmic scaling)
    try:
        followers_weight = (1 + (current_user.followerCount / 1000)) / (1 + (other_user.followerCount / 1000))
    except (ValueError, TypeError):
        followers_weight = 0  
    similarity_score += followers_weight * 10  # Small weight for followers

    # Adding a random variability factor to diversify scores
    similarity_score += random.uniform(-2, 2)  # Small random variability

    return similarity_score

# Endpoint to calculate and return recommendations
@app.post("/recommend_users")
async def recommend_users(input_data: InputData):
    current_user = input_data.currentUser
    other_users = input_data.otherUsers

    similarity_scores = []

    for other_user in other_users:
        if current_user.id == other_user.id:
            continue

        similarity_score = calculate_similarity(current_user, other_user, openai_api_key)
        similarity_scores.append((other_user.id, similarity_score))

    sorted_similarities = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    return {
        "input_user": current_user.id,
        "recommended_users": [{"user_id": user[0], "similarity_score": user[1]} for user in sorted_similarities[:10]]
    }

# To run the app, use: uvicorn main:app --reload
