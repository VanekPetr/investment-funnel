from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # Would you like to download the latest AlgoStrata's data?
    # TODO write to kourosh@algostrata.dk to get your own API key
    NAME: str = "Investment Funnel Secrets"
    ALGOSTRATA_NAMES_URL: str = "private"
    ALGOSTRATA_PRICES_URL: str = "notgoingtotellyou"
    ALGOSTRATA_KEY: str = "nonono"


settings = Settings()
