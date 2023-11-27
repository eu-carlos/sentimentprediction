# Importando Flask e Bibliotecas
from flask import Flask, render_template, request
import pickle
import nltk

nltk.download("rslp")

app = Flask(__name__)

# Carregar o modelo salvo em PKL que foi treina
with open('pipe.pkl', 'rb') as file:
    classificador = pickle.load(file)

# Função para extrair características
def extrator_palavras(text):
    stemmer = nltk.stem.RSLPStemmer()
    tokens = [str(stemmer.stem(palavra.lower())) for palavra in text.split()]
    return dict([(token, True) for token in tokens])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Criando um formulário para que seja inserido o texto
        text = request.form['text']

        # Fazendo a previsão do sentimento do texto com o modelo de ML
        features = extrator_palavras(text)
        result = classificador.classify(features)

        return render_template('index.html', prediction=result, input_text=text)

# rodar o app e ativar o debug
if __name__ == '__main__':
    app.run(debug=False)

# é possível iniciá-lo pelo terminal com python app.py
