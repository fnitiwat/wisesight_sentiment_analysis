from fastapi import FastAPI
from starlette.status import HTTP_200_OK
from starlette.middleware.cors import CORSMiddleware

from schema import APIRequestBody, APIResponseBody
from predictor import predictor


app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post(
    "/predict",
    summary="Sentiment Analysis",
    response_model=APIResponseBody,
    status_code=HTTP_200_OK,
)
def predict(request_body: APIRequestBody):
    text = request_body.text
    sentiment = predictor.predict(text)
    return {"sentiment": sentiment}
