from pydantic import BaseModel


class APIRequestBody(BaseModel):
    text: str


class APIResponseBody(BaseModel):
    sentiment: str
