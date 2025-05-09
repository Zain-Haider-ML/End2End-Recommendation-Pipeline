# Variables
PYTHON=python
PIP=pip
APP_NAME_STREAMLIT=streamlit-app
APP_NAME_FLASK=flask-api

# Install all dependencies
install:
	$(PIP) install -r requirements.txt

# Lint the code using flake8
lint:
	flake8 --ignore=E203,E501,W605,F401 app.py api.py models.py

# # Run tests
# test:
# 	pytest tests/

prediction:
	bash make_prediction.sh

# Run Streamlit app locally
run-streamlit:
	streamlit run app.py

# Run Flask API locally
run-flask:
	$(PYTHON) api.py

# Build Docker image for Streamlit
docker-build-streamlit:
	docker build -t $(APP_NAME_STREAMLIT) -f Dockerfile.streamlit .

# Run Docker container for Streamlit
docker-run-streamlit:
	docker run -p 8501:8501 $(APP_NAME_STREAMLIT)

# Build Docker image for Flask
docker-build-flask:
	docker build -t $(APP_NAME_FLASK) -f Dockerfile.flask .

# Run Docker container for Flask
docker-run-flask:
	docker run -p 5000:5000 $(APP_NAME_FLASK)

# Clean Docker images
docker-clean:
	docker rmi $(APP_NAME_STREAMLIT) $(APP_NAME_FLASK) || true

# Format the code using black
format:
	black app.py api.py models.py

# Help command
help:
	@echo "Makefile Commands:"
	@echo "  install               Install dependencies"
	@echo "  lint                  Lint the code with flake8"
	@echo "  test                  Run unit tests with pytest"
	@echo "  run-streamlit         Run the Streamlit app"
	@echo "  run-flask             Run the Flask API"
	@echo "  docker-build-streamlit  Build Streamlit Docker image"
	@echo "  docker-run-streamlit    Run Streamlit Docker container"
	@echo "  docker-build-flask     Build Flask Docker image"
	@echo "  docker-run-flask       Run Flask Docker container"
	@echo "  docker-clean           Remove Docker images"
	@echo "  format                Format the code with black"
	@echo "  help                  Show this help"
