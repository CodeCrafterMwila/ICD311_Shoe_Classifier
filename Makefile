.PHONY: setup run clean

setup:
	@echo "Creating virtual environment..."
	python -m venv venv
	@echo "Installing dependencies..."
	@if [ -d "venv/Scripts" ]; then \
		venv\Scripts\pip install --upgrade pip; \
		venv\Scripts\pip install -r requirements.txt; \
	else \
		venv/bin/pip install --upgrade pip; \
		venv/bin/pip install -r requirements.txt; \
	fi
	@echo "Setup complete! Run 'make run' to launch the app."

run:
	@echo "Launching Streamlit Shoe Classifier..."
	@if [ -d "venv/Scripts" ]; then \
		venv\Scripts\streamlit run app.py; \
	else \
		venv/bin/streamlit run app.py; \
	fi

clean:
	@echo "Cleaning up cache and virtual environment..."
	rm -rf __pycache__ .pytest_cache venv
	find . -type f -name "*.pyc" -delete