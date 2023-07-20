from flask import Flask, request, jsonify
import main

app = Flask(__name__)

@app.route('/generate_flashcards', methods=['POST'])
def generate_flashcards():
    # Get the text input and number of flashcards from the request
    text = request.json['text']
    num_flashcards = request.json['num_flashcards_limit']

    # Call your machine learning model to generate the flashcards
    flashcards = main.generate_flashcards(text, num_flashcards)

    # Return the flashcards as a JSON response
    response = {
        'flashcards': flashcards
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run()