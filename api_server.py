from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from itinerary_generator import ItineraryGenerator
import uvicorn

app = FastAPI(title="AI Itinerary Generator API")

# Initialize the model once when the server starts
generator = ItineraryGenerator()

class ItineraryRequest(BaseModel):
    days: str
    city: str
    traveler_type: str
    budget: str

class ItineraryResponse(BaseModel):
    itinerary: str

@app.post("/generate-itinerary", response_model=ItineraryResponse)
async def generate_itinerary(request: ItineraryRequest):
    try:
        result = generator.generate_itinerary(
            days=request.days,
            city=request.city,
            traveler_type=request.traveler_type,
            budget=request.budget
        )
        return ItineraryResponse(itinerary=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating itinerary: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)