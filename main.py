from flask import Flask, redirect, url_for, render_template, request, jsonify

from server.resume import answer_query, warm_up

app = Flask(__name__)

# Initialize RAG and build index when the app starts (not on first query)
warm_up()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/ask', methods=['POST'])
def api_ask():
    """Receive the text input from the main page and return the answer from resume module."""
    data = request.get_json(silent=True) or {}
    query = data.get('query', '')
    answer = answer_query(query)
    return jsonify({'answer': answer})


if __name__ == '__main__':
    #app.run(debug=True)
    print("App running in production mode")
