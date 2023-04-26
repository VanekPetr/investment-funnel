from pydantic import BaseSettings
from dotenv import load_dotenv
load_dotenv()


class Settings(BaseSettings):
    NAME: str = "Investment Funnel Secrets"
    ALGOSTRATA_NAMES_URL: str = "private"
    ALGOSTRATA_PRICES_URL: str = "notgoingtotellyou"
    ALGOSTRATA_KEY: str = "nonono"


settings = Settings()
