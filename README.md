# Flashcard Generator

The Flashcard Generator is a project that utilizes the BART model to automatically generate flashcards from a given text. It provides an API endpoint to generate flashcards based on user input. The project leverages natural language processing techniques and deep learning models to extract relevant information and generate high-quality flashcards.

## Features

- Automatic Flashcard Generation: The BART model and transformers library are used to generate questions and answers based on the input text. The generated flashcards capture key information from the text, making it easy for users to study and review.

- Question Ranking: The generated questions are ranked based on their quality and relevance using the Sentence-BERT model. This ensures that the most significant questions are selected for the flashcards, enhancing the learning experience.

- Question Answering: The flashcards include both the question and the corresponding answer. The Question Answering (QA) pipeline is utilized to extract accurate answers from the input text, providing comprehensive and reliable information.

## Requirements

- nltk: The Natural Language Toolkit (NLTK) is used for text processing and tokenization.
- transformers: The transformers library is required to load the BART model and tokenizer.
- torch: PyTorch is used as the deep learning framework.
- sentence-transformers: The SentenceTransformer library is utilized to encode questions and compute similarity scores.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/flashcard-generator.git
```

2. Install the required dependencies:
```python
pip install nltk transformers torch sentence-transformers
```

3. Download the NLTK resources:
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
```

4. Obtain the BART model and Sentence-BERT model:
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

tokenizer = AutoTokenizer.from_pretrained("voidful/bart-eqg-question-generator")
model = AutoModelForSeq2SeqLM.from_pretrained("voidful/bart-eqg-question-generator")
sentence_bert_model = SentenceTransformer('average_word_embeddings_glove.6B.300d')
```
## API Usage
The Flashcard Generator provides a RESTful API endpoint to generate flashcards. Send a POST request to the following URL:
```bash
http://localhost:5000/generate_flashcards
```

Request Body:
```json
{
  "num_flashcards": 5,
  "text": "Enter your text here..."
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

## Example Usage
```python
import requests

url = "http://localhost:5000/generate_flashcards"
data = {
  "text": "Enter your text here...",
  "num_flashcards": 5
}

response = requests.post(url, json=data)
flashcards = response.json()["flashcards"]

for flashcard in flashcards:
    print("Question:", flashcard["question"])
    print("Answer:", flashcard["answer"])
    print()
```

## In Progress
I am actively working on implementing parallel computing techniques to enhance the performance of the flashcard generation process. By leveraging parallelization, my aim is to improve the efficiency and reduce the processing time, allowing for faster generation of flashcards from larger texts. 

In addition to implementing parallel computing, I am planning to incorporate user feedback functionality into the project. This feature will allow users to receive real-time feedback on their performance and progress, enabling them to assess their learning and track their improvement over time.



## License
This project is licensed under the MIT License.
