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

- `.env`: This file stores environment variables such as API keys (e.g., for OpenAI or Google Generative AI) and other sensitive configurations. It's crucial for keeping credentials out of the codebase.
- `generatechunks.py`: This script is responsible for taking raw video transcripts, breaking them down into smaller, manageable "chunks," and then embedding these chunks into a vector database. This process is vital for efficient retrieval of relevant information.
- `retrieval.py`: This script handles the core retrieval-augmented generation (RAG) logic. It queries the vector database with a user's question, retrieves the most relevant transcript chunks, and then uses a language model to generate a coherent answer based on the retrieved context.
- `serviceAccountKey.json`: This JSON file contains the credentials for Firebase Admin SDK, enabling the application to interact with Firebase services (e.g., Firestore for data storage, Authentication for user management).
- `storetranscript.py`: This script is dedicated to fetching transcripts from YouTube videos using the `youtube-transcript-api` library and storing them, potentially in the `documents/` directory or directly processing them for the vector database.
- `streamlitapp.py`: This is the main entry point for the user interface. It's a Streamlit application that provides a web-based interface for users to input YouTube URLs, trigger transcript processing, and interact with the chatbot.
- `db/`: This directory houses the ChromaDB vector database. It contains the necessary files for storing and indexing the embedded transcript chunks, allowing for fast semantic searches.
- `documents/`: This directory is intended for storing raw transcript files or other textual documents that the chatbot might process.
- `virt/`: This directory contains the Python virtual environment, which isolates project dependencies from the system-wide Python installation.

## Setup and Installation

Follow these steps to set up and run the project locally:

### 1. Clone the Repository

```bash
git clone https://github.com/aman081/YT-BOT.git
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

Install all required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file in the root directory of the project and add your API keys and other necessary configurations.

```
# Example .env content
OPENAI_API_KEY="your_openai_api_key_here"
GOOGLE_API_KEY="your_google_generative_ai_key_here"
# Add any other API keys or configurations required by the project
```

### 6. Set up Firebase (if applicable)

Place your `serviceAccountKey.json` file in the root directory of the project. This file is crucial for Firebase authentication and interaction.

### 7. Run the Streamlit Application

Once all dependencies are installed and configurations are set, you can run the Streamlit app:

```bash
streamlit run streamlitapp.py
```

This will open the application in your web browser, typically at `http://localhost:8501`.

## Usage

1. **Enter YouTube Video URL**: In the Streamlit application's sidebar or input field, paste the URL of the YouTube video you wish to analyze.
2. **Fetch and Process Transcript**: Click the "Fetch Transcript" or similar button. The application will then:
    - Download the video's transcript.
    - Process the transcript into smaller, semantically relevant chunks.
    - Embed these chunks and store them in the ChromaDB vector database.
3. **Ask Questions**: Once the transcript is processed, a chat interface will appear. You can type your questions related to the video content into the input box.
4. **Receive Responses**: The chatbot will use the RAG mechanism to retrieve relevant information from the stored transcript chunks and generate a concise, context-aware answer to your question.

## Contributing

We welcome contributions to improve this project! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Make your changes and ensure the code adheres to the project's style.
4. Write clear commit messages.
5. Push your branch (`git push origin feature/YourFeature`).
6. Open a Pull Request with a detailed description of your changes.

## License

This project is open-source and available under the [MIT License](LICENSE).
