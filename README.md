# YT-CHATBOT

This project is a YouTube Chatbot that allows users to interact with YouTube video transcripts. It leverages various Python libraries and services to fetch transcripts, process them into manageable chunks, store them in a vector database, and retrieve relevant information based on user queries.

## Features

- **Transcript Fetching**: Automatically fetches transcripts from YouTube videos.
- **Chunking**: Divides long transcripts into smaller, semantically meaningful chunks for efficient processing.
- **Vector Database Integration**: Stores transcript chunks in a ChromaDB vector database for fast similarity searches.
- **Retrieval-Augmented Generation (RAG)**: Uses a retrieval mechanism to find relevant transcript sections and then generates responses based on these sections.
- **Streamlit UI**: Provides an interactive web interface using Streamlit for easy interaction.
- **Firebase Integration**: (Assumed based on `serviceAccountKey.json`) Potentially used for user authentication, data storage, or other backend services.

## Project Structure

- `.env`: Environment variables for API keys and other configurations.
- `generatechunks.py`: Script for processing transcripts and generating chunks.
- `retrieval.py`: Script for retrieving information from the vector database.
- `serviceAccountKey.json`: Firebase service account key for authentication.
- `storetranscript.py`: Script for fetching and storing YouTube video transcripts.
- `streamlitapp.py`: The main Streamlit application file.
- `db/`: Directory containing the ChromaDB vector database files.
- `documents/`: Directory for storing raw transcript files.
- `virt/`: Virtual environment directory.

## Setup and Installation

Follow these steps to set up and run the project locally:

### 1. Clone the Repository

```bash
git clone https://github.com/satyam969/YT-BOT.git
cd YT-BOT
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python -m venv virt
```

### 3. Activate the Virtual Environment

- **On Windows:**
  ```bash
  .\virt\Scripts\activate
  ```
- **On macOS/Linux:**
  ```bash
  source virt/bin/activate
  ```

### 4. Install Dependencies

Install all required Python packages. You might need a `requirements.txt` file for this. If not available, you'll need to install them manually. Assuming common libraries:

```bash
pip install -r requirements.txt
# If requirements.txt is not present, you might need to install:
# pip install youtube-transcript-api chromadb streamlit python-dotenv firebase-admin
```
*(Note: A `requirements.txt` file is not present in the current directory. You may need to create one based on the project's dependencies.)*

### 5. Configure Environment Variables

Create a `.env` file in the root directory of the project and add your API keys and other necessary configurations.

```
# Example .env content
OPENAI_API_KEY="your_openai_api_key_here"
# Add any other API keys or configurations required by the project
```

### 6. Set up Firebase (if applicable)

Place your `serviceAccountKey.json` file in the root directory of the project. This file is crucial for Firebase authentication.

### 7. Run the Streamlit Application

Once all dependencies are installed and configurations are set, you can run the Streamlit app:

```bash
streamlit run streamlitapp.py
```

This will open the application in your web browser.

## Usage

1. **Enter YouTube Video URL**: In the Streamlit application, provide the URL of the YouTube video you want to analyze.
2. **Fetch and Process Transcript**: The application will fetch the transcript, chunk it, and store it in the vector database.
3. **Ask Questions**: You can then ask questions related to the video content, and the chatbot will retrieve relevant information and generate responses.

## Contributing

Feel free to fork the repository, make improvements, and submit pull requests.

## License

This project is open-source and available under the [MIT License](LICENSE).
