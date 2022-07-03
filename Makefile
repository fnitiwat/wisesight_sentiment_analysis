start_app:
	cd app/ && python3 -m uvicorn main:app --host=0.0.0.0 --port=8000 --reload

docker_build:
	docker build -t wisesight_sentiment_analysis .

docker_run:
	docker run -p 8000:8000 wisesight_sentiment_analysis

docker_run_cpu:
	docker run -p 8000:8000 -e DEVICE="cpu" wisesight_sentiment_analysis

docker_push:
	docker tag e475fcc45766 498150577381.dkr.ecr.ap-southeast-1.amazonaws.com/sentiment-analysis && \
	docker push 498150577381.dkr.ecr.ap-southeast-1.amazonaws.com/sentiment-analysis
