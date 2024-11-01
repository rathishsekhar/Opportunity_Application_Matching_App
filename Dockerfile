# Use the official Python image
FROM python:3.12-bookworm

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

# Set the working directory inside the container
WORKDIR $APP_HOME

# Copy requirements.txt and install dependencies
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the app code into the container
COPY . .

# Expose the port on which Streamlit runs
EXPOSE 8080

# Command to run the Streamlit app
CMD streamlit run app.py --server.port 8080 --server.enableCORS false
