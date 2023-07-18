import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import ipywidgets as widgets
from IPython.display import display, clear_output
from sentence_transformers import SentenceTransformer


# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load BART model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("voidful/bart-eqg-question-generator")
model = AutoModelForSeq2SeqLM.from_pretrained("voidful/bart-eqg-question-generator")
qa_model = pipeline("question-answering")

# Load Sentence-BERT model
sentence_bert_model = SentenceTransformer('average_word_embeddings_glove.6B.300d')


# Define the event handler for the button click
def generate_flashcards(text, num_flashcards):      
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Remove stopwords and perform lemmatization for each sentence
    stopwords_set = set(nltk_stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    filtered_sentences = []
    for sentence in sentences:
        tokens = word_tokenize(sentence.lower())
        filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords_set]
        filtered_sentence = " ".join(filtered_tokens)
        filtered_sentences.append(filtered_sentence)

    # Generate questions
    questions = []
    for sentence in filtered_sentences:
        input_ids = tokenizer.encode(sentence, add_special_tokens=True, truncation=True, max_length=512, padding='max_length')
        input_ids = input_ids[:512]  # Limit the input length for BART
        input_ids = torch.tensor(input_ids).unsqueeze(0)

        question_ids = model.generate(input_ids, num_beams=4, min_length=10, max_length=150, length_penalty=2.0)
        question = tokenizer.decode(question_ids[0], skip_special_tokens=True)
        questions.append(question)

    # Rank questions based on quality/relevance
    question_embeddings = sentence_bert_model.encode(questions)
    ranking_scores = question_embeddings.dot(question_embeddings.T).mean(axis=1)  # Compute cosine similarity scores

    # Filter and select the most significant questions
    selected_indices = ranking_scores.argsort()[-num_flashcards:][::-1]
    selected_questions = [questions[i] for i in selected_indices]

    answers = []
    for question in selected_questions:
        context = text
        answer = qa_model(question = question, context = context)
        answers.append(answer)

    
    flashcards = []
    for i in range(len(selected_questions)):
      flashcard = {
          'question': questions[i],
          'answer': answers[i]
      }
      flashcards.append(flashcard)

    return flashcards

