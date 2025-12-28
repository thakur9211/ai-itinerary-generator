from itinerary_generator import ItineraryGenerator

# Initialize the generator
generator = ItineraryGenerator()

# Test parameters
days = "2 days"
city = "Noida"
traveler_type = "one person"
budget = "500"

# Generate itinerary
result = generator.generate_itinerary(days, city, traveler_type, budget)

print(f"\n--- GENERATED {city.upper()} ITINERARY ---\n")
print(result)