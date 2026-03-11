import numpy as np

# -----------------------------
# Função Softmax
# -----------------------------
def softmax(x):
    exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)


# -----------------------------
# Scaled Dot Product Attention
# -----------------------------
def self_attention(X):

    d_model = X.shape[-1]

    # Matrizes de projeção
    Wq = np.random.randn(d_model, d_model)
    Wk = np.random.randn(d_model, d_model)
    Wv = np.random.randn(d_model, d_model)

    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    # Produto escalar
    scores = (Q @ K.transpose(0,2,1)) / np.sqrt(d_model)

    # Atenção
    attn = softmax(scores)

    # Saída
    output = attn @ V

    return output


# -----------------------------
# Layer Normalization
# -----------------------------
def layer_norm(X, eps=1e-6):

    mean = np.mean(X, axis=-1, keepdims=True)
    var = np.var(X, axis=-1, keepdims=True)

    return (X - mean) / np.sqrt(var + eps)


# -----------------------------
# Feed Forward Network
# -----------------------------
def ffn(X):

    d_model = X.shape[-1]
    d_ff = 256

    W1 = np.random.randn(d_model, d_ff)
    b1 = np.random.randn(d_ff)

    W2 = np.random.randn(d_ff, d_model)
    b2 = np.random.randn(d_model)

    hidden = np.maximum(0, X @ W1 + b1)  # ReLU
    output = hidden @ W2 + b2

    return output


# -----------------------------
# Encoder Layer
# -----------------------------
def encoder_layer(X):

    # Self Attention
    attn_output = self_attention(X)

    # Residual + Normalization
    X1 = layer_norm(X + attn_output)

    # Feed Forward
    ffn_output = ffn(X1)

    # Residual + Normalization
    X2 = layer_norm(X1 + ffn_output)

    return X2


# -----------------------------
# Transformer Encoder (6 camadas)
# -----------------------------
def transformer_encoder(X, num_layers=6):

    output = X

    for i in range(num_layers):
        output = encoder_layer(output)

    return output


# -----------------------------
# Exemplo de uso
# -----------------------------

# Vocabulário simples
vocab = {
    "o":0,
    "banco":1,
    "bloqueou":2,
    "cartao":3
}

frase = ["o","banco","bloqueou","cartao"]

ids = [vocab[p] for p in frase]

vocab_size = len(vocab)
d_model = 64

# Embedding aleatório
embedding_table = np.random.randn(vocab_size, d_model)

X = embedding_table[ids]

# adicionar dimensão batch
X = np.expand_dims(X, axis=0)

# Rodar encoder
Z = transformer_encoder(X)

print("Shape da saída:", Z.shape)
