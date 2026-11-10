"""Word lists for the generator. Business names only (no person names or phones, CLAUDE.md rule 1).

shortcut: hand-written lists; OSM names for the chosen cities replace PLACE_NAMES / LANDMARKS in step 8.
"""
import random

PLACE_NAMES = [
    "Shree Sweets", "New Bharat Bakery", "Royal Medical Store", "Sai Mobile Shop", "Hotel Rajdhani",
    "Vijay Electronics", "Ganesh Stationery", "Shanti Dental Clinic", "Apna Kirana Store", "Balaji Tyres",
    "Krishna Book Depot", "Laxmi Jewellers", "Om Hardware", "Annapurna Mess", "Raj Tailors",
    "City Super Market", "Maa Durga Photo Studio", "Jai Hind Cycle Store", "Sunrise Coaching Classes",
    "Lotus Beauty Parlour", "Green Park Nursery", "Anand Fruit Centre", "Mahalaxmi Textiles",
    "Bharat Petrol Pump", "Sanjeevani Hospital", "Little Flower School", "Government Primary School",
    "Hanuman Mandir", "Jama Masjid", "St. Mary's Church", "District Library", "Post Office",
]
HOUSE_WORDS = ["Flat", "House No.", "Plot No.", "Shop No.", "Door No.", "Room No.", "Office No.", "Gala No.", "Plot", "Shop"]
FLOORS = ["Ground Floor", "1st Floor", "2nd Floor", "3rd Floor", "4th Floor"]
BUILDING_NAMES = [
    "Shanti", "Sunshine", "Green Park", "Sai", "Krishna", "Ganga", "Shivam", "Surya", "Chandra", "Kaveri",
    "Lotus", "Pearl", "Silver Oak", "Mahaveer", "Rajat", "Vasant", "Anand", "Nirmal", "Prestige", "Galaxy",
]
BUILDING_TYPES = ["Apartments", "Society", "Complex", "Enclave", "Residency", "Towers", "Heights", "Plaza",
                  "Chambers", "Arcade", "Bhavan", "Niwas"]
STREET_NAMES = [
    "MG", "Station", "Nehru", "Gandhi", "Link", "Ring", "College", "Temple", "Market", "Hospital", "Mall",
    "Church", "Park", "Jail", "Court", "Bazar", "Mandir", "Main", "Old Bus Stand", "Subhash", "Tilak",
    "Ambedkar", "Bose", "Patel", "Rajiv", "Indira", "Shastri", "Airport", "Industrial", "Canal",
]
STREET_TYPES = ["Road", "Marg", "Street", "Lane", "Path"]
LANDMARKS = [
    "Hanuman Mandir", "Shiv Mandir", "Durga Mandir", "Sai Baba Mandir", "Jama Masjid", "Gurudwara", "Church",
    "SBI ATM", "SBI Bank", "HDFC Bank", "Bank of Baroda", "Bus Stand", "Railway Station", "Petrol Pump",
    "Government School", "Government Hospital", "Primary Health Centre", "Water Tank", "Police Station",
    "Post Office", "Panchayat Bhavan", "Anganwadi Centre", "Over Bridge", "Railway Crossing", "Chowk",
    "Clock Tower", "Vegetable Market", "Cinema Hall", "Community Hall", "Big Bazaar", "Reliance Fresh",
    "Apollo Pharmacy", "Axis Bank", "Canara Bank", "Court", "Collectorate", "Stadium", "Lake", "Park",
]


def house(rng: random.Random) -> str:
    if rng.random() < 0.2:
        return rng.choice(FLOORS)
    number = str(rng.randint(1, 250)) + (rng.choice(["", "", "A", "B", "/1", "/2", "-3"]))
    return f"{rng.choice(HOUSE_WORDS)} {number}"


def building(rng: random.Random) -> str:
    return f"{rng.choice(BUILDING_NAMES)} {rng.choice(BUILDING_TYPES)}"


def street(rng: random.Random, place: str) -> str:
    r = rng.random()
    if r < 0.15:
        return f"{ordinal(rng.randint(1, 12))} Cross"
    if r < 0.3:
        return f"{ordinal(rng.randint(1, 12))} Main"
    if r < 0.4:
        return f"Gali No. {rng.randint(1, 20)}"
    if r < 0.45:
        return f"NH {rng.randint(1, 99)}"
    # half the named streets take a real place name: with 30 fixed names a model memorises streets instead of reading "Road"
    name = place if rng.random() < 0.5 else rng.choice(STREET_NAMES)
    return f"{name} {rng.choice(STREET_TYPES)}"


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"
