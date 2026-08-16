from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URL = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DB_NAME_USER")

client = AsyncIOMotorClient(MONGO_URL)

database = client[DATABASE_NAME]

# collection ví dụ
users_collection = database["users"]