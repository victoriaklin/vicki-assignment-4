from flask import Flask, request, render_template, jsonify
from lsa import retrieve_documents

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        results = retrieve_documents(query)
        return render_template('index.html', query=query, results=results)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)