from dotenv import load_dotenv
from pathlib import Path

env_path = (
    Path(__file__)
    .resolve()
    .parent / '.env'
)

load_dotenv(env_path)