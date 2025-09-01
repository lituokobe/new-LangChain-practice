from dotenv import load_dotenv
import os

load_dotenv(override = True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

ALI_API_KEY = os.getenv("ALI_API_KEY")
ALI_BASE_URL = os.getenv("ALI_BASE_URL")

