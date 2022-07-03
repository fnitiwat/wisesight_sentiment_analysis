start_app:
	cd app/ && python3 -m uvicorn main:app --host=0.0.0.0 --port=8000 --reload