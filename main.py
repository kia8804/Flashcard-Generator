import torch
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load BART model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("valhalla/bart-large-finetuned-squadv1")
model = AutoModelForQuestionAnswering.from_pretrained("valhalla/bart-large-finetuned-squadv1")

pipe = pipeline("text2text-generation", 'lmqg/t5-large-squad-qg')

# Load Sentence-BERT model
sentence_bert_model = SentenceTransformer('average_word_embeddings_glove.6B.300d')

# Define the event handler for the button click
def generate_flashcards(text, num_flashcards):      
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Generate questions
    questions = []
    for sentence in sentences:
        input_text = 'generate question: ' + sentence
        question = pipe(input_text)
        question = question[0]['generated_text']
        questions.append(question)

    # Rank questions based on quality/relevance
    question_embeddings = sentence_bert_model.encode(questions)
    ranking_scores = question_embeddings.dot(question_embeddings.T).mean(axis=1)  # Compute cosine similarity scores

    # Filter and select the most significant questions
    selected_indices = ranking_scores.argsort()[-num_flashcards:][::-1]
    selected_questions = [questions[i] for i in selected_indices]

    answers = []
    for question in selected_questions:
        inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]

        text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        
        outputs = model(**inputs)
        answer_start_scores=outputs.start_logits
        answer_end_scores=outputs.end_logits

        answer_start = torch.argmax(
            answer_start_scores
        )  # Get the most likely beginning of answer with the argmax of the score
        answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

        # Combine the tokens in the answer and print it out.""
        answer = answer.replace("#","")

        answers.append(answer)

    
    flashcards = []
    for i in range(len(selected_questions)):
      flashcard = {
          'question': selected_questions[i],
          'answer': answers[i]
      }
      flashcards.append(flashcard)

    return flashcards

