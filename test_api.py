import requests
import json

# API endpoint
url = "http://192.168.0.169:8001/generate-itinerary"

# Test data
test_data = {
    "days": "2 days",
    "city": "Jaipur",
    "traveler_type": "one person",
    "budget": "1500"
}

try:
    response = requests.post(url, json=test_data)
    
    if response.status_code == 200:
        result = response.json()
        print("Generated Itinerary:")
        print("=" * 50)
        print(result["itinerary"])
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        
except requests.exceptions.ConnectionError:
    print("Error: Could not connect to the API server. Make sure it's running on localhost:8000")
except Exception as e:
    print(f"Error: {e}")