from flask import Flask, render_template, request, jsonify
from transformers import DistilBertTokenizer, TFDistilBertForQuestionAnswering, GPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf

app = Flask(__name__)

# Load pre-trained DistilBERT model and tokenizer for Question Answering
distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
distilbert_model = TFDistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

# Load pre-trained GPT-2 model and tokenizer for Next Word Prediction
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Sample text for GPT-2 Next Word Prediction
with open('data.txt', 'r') as file:
    gpt2_text = file.read()

@app.route('/')
def index():
    return render_template('index_combined.html')

@app.route('/qa_predict', methods=['POST'])
def qa_predict():
    if request.method == 'POST':
        question = request.form['question']

        inputs = distilbert_tokenizer(question, gpt2_text, return_tensors="tf")
        outputs = distilbert_model(**inputs)

        answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
        answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])

        predict_answer_tokens = inputs.input_ids[0, answer_start_index:answer_end_index + 1]
        predicted_answer = distilbert_tokenizer.decode(predict_answer_tokens)

        return render_template('index_combined.html', qa_question=question, qa_answer=predicted_answer, nw_input="", nw_prediction="")

@app.route('/nw_predict', methods=['POST'])
def nw_predict():
    if request.method == 'POST':
        input_sentence = request.form['input_sentence']
        predicted_words = predict_next_words(input_sentence)
        return render_template('index_combined.html', qa_question="", qa_answer="", nw_input=input_sentence, nw_prediction=predicted_words)

def predict_next_words(sentence):
    input_ids = gpt2_tokenizer.encode(sentence, return_tensors="pt")
    output = gpt2_model.generate(input_ids, max_length=50, num_return_sequences=1)
    generated_text = gpt2_tokenizer.decode(output[0, input_ids.shape[-1]:], skip_special_tokens=True)
    next_words = " ".join(generated_text.split()[:4])
    return next_words

if __name__ == '__main__':
    app.run(debug=True)
