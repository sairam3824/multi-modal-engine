.PHONY: install test run-api run-demo clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

run-api:
	uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

run-demo:
	streamlit run src/demo.py

clean:
	rm -rf __pycache__ .pytest_cache uploads/
	find . -type d -name "__pycache__" -exec rm -rf {} +
