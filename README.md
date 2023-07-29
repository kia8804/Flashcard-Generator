# Flashcard Generator

The Flashcard Generator is a project that utilizes the BART model to automatically generate flashcards from a given text. It provides an API endpoint to generate flashcards based on user input. The project leverages natural language processing techniques and deep learning models to extract relevant information and generate high-quality flashcards.

## How It Works

### Main Functionality (main.py)

The core functionality of the Flashcard/Question Answering system is implemented in `main.py`. The file utilizes pre-trained models to accomplish the following tasks:

1. Tokenize the input text into sentences using the NLTK library.
2. Generate questions from each sentence using the `text2text-generation` pipeline powered by T5-based language models.
3. Rank the generated questions based on their relevance using Sentence-BERT, a sentence embedding model.
4. Select the most significant questions based on the ranking.
5. Utilize the BART-based model to find accurate answers to the selected questions within the input text.
6. Create a set of flashcards, each containing a generated question and its corresponding answer.

### API (app.py)

The API is created using the Flask web framework and resides in the `app.py` file. It sets up an HTTP server that exposes a single endpoint:

- `/generate_flashcards`: Accepts POST requests with a JSON payload containing the `text` and `num_flashcards_limit` parameters. The `text` parameter should contain the text from which flashcards need to be generated, and `num_flashcards_limit` specifies the maximum number of flashcards to create.

Upon receiving the request, the API calls the `generate_flashcards` function from `main.py`, passing the input text and the limit for the number of flashcards. The resulting flashcards are then returned as a JSON response.


## Requirements

- nltk: The Natural Language Toolkit (NLTK) is used for text processing and tokenization.
- transformers: The transformers library is required to load the BART model and tokenizer.
- torch: PyTorch is used as the deep learning framework.
- sentence-transformers: The SentenceTransformer library is utilized to encode questions and compute similarity scores.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/Flashcard-Generator.git
```

2. Install the required dependencies:
```python
pip install flask transformers sentence-transformers nltk
```

## API Usage
Run the API file:
```python
python app.py
```

The Flashcard Generator provides a RESTful API endpoint to generate flashcards. Send a POST request to the following URL:
```bash
http://localhost:5000/generate_flashcards
```

Replace "Enter your text here." with the text you want to generate flashcards from and set "num_flashcards_limit" to the maximum number of flashcards you want to create in the request body:
:
```json
{
  "num_flashcards_limit": 5,
  "text": "Enter your text here."
}
```

- "text" (string): The input text from which flashcards will be generated.
- "num_flashcards" (integer): The number of flashcards to generate.
  
The API will respond with a JSON object containing the generated flashcards:
```json
{
  "flashcards": [
    {
      "question": "Question 1",
      "answer": "Answer 1"
    },
    {
      "question": "Question 2",
      "answer": "Answer 2"
    },
      "question": "...",
      "answer": "..." 
  ]
}
```

### In Progress
I am actively working on implementing parallel computing techniques to enhance the performance of the flashcard generation process. By leveraging parallelization, my aim is to improve efficiency and reduce the processing time, allowing for faster generation of flashcards from larger texts. 

In addition to implementing parallel computing, I am planning to incorporate user feedback functionality into the project. This feature will allow users to receive real-time feedback on their performance and progress, enabling them to assess their learning and track their improvement over time.



## License
This project is licensed under the MIT License.
