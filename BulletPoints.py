import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Load BART model and tokenizer
model_name = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Sample large piece of text
corpus = " "

# Tokenize the text into sentences
sentences = sent_tokenize(corpus)

# Remove stopwords and perform lemmatization for each sentence
stopwords = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
filtered_sentences = []
for sentence in sentences:
    tokens = word_tokenize(sentence.lower())
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords]
    filtered_sentence = " ".join(filtered_tokens)
    filtered_sentences.append(filtered_sentence)

# Generate bullet points
bullet_points = []
for sentence in filtered_sentences:
    input_ids = tokenizer.encode(sentence, add_special_tokens=True, truncation=True, max_length=512, padding='max_length')
    input_ids = input_ids[:512]  # Limit the input length for BART
    input_ids = torch.tensor(input_ids).unsqueeze(0)

    summary_ids = model.generate(input_ids, num_beams=4, min_length=10, max_length=150, length_penalty=2.0)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    bullet_points.append(summary.strip())

# Print bullet points
if not bullet_points:
    print("No bullet points found.")
else:
    print("Bullet Points:")
    for point in bullet_points:
        print("- " + point)
