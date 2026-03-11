
---

# Transformer Encoder From Scratch

Este projeto implementa uma versão **simplificada do Transformer Encoder** utilizando apenas **Python e NumPy**, sem o uso de frameworks de deep learning como TensorFlow ou PyTorch.

O objetivo é demonstrar os principais componentes da arquitetura apresentada no artigo:

**"Attention Is All You Need" (Vaswani et al., 2017).**

---

# Estrutura do Projeto

O código implementa os seguintes componentes fundamentais do **Transformer Encoder**:

1. **Embedding**
2. **Self-Attention**
3. **Scaled Dot Product Attention**
4. **Layer Normalization**
5. **Feed Forward Network**
6. **Residual Connections**
7. **Empilhamento de múltiplas camadas do Encoder**

---

# Tecnologias Utilizadas

* Python 3
* NumPy
* Pandas (opcional para manipulação de vocabulário)

Instalação das dependências:

```bash
pip install numpy pandas
```

---

# Funcionamento do Código

## 1. Criação do Vocabulário

Primeiro é criado um vocabulário simples que associa cada palavra a um índice.

Exemplo:

```python
vocab = {
    "o":0,
    "banco":1,
    "bloqueou":2,
    "cartao":3
}
```

Uma frase é convertida em **IDs numéricos** para ser utilizada pelo modelo.

---

# 2. Embedding

Cada token é transformado em um vetor denso chamado **embedding**.

```python
embedding_table = np.random.randn(vocab_size, d_model)
X = embedding_table[ids]
```

Dimensões:

```
(batch_size, sequence_length, d_model)
```

---

# 3. Self-Attention

O mecanismo de **Self-Attention** permite que cada palavra da frase preste atenção nas outras palavras da mesma frase.

São calculadas três projeções:

* Query (Q)
* Key (K)
* Value (V)

```python
Q = X @ Wq
K = X @ Wk
V = X @ Wv
```

---

# 4. Scaled Dot Product Attention

O score de atenção é calculado usando:

```
Attention(Q,K,V) = softmax(QKᵀ / √d) V
```

Implementação:

```python
scores = (Q @ K.transpose(0,2,1)) / np.sqrt(d_model)
attn = softmax(scores)
output = attn @ V
```

---

# 5. Residual Connection + Layer Normalization

Para melhorar a estabilidade do treinamento, o Transformer utiliza:

* **Conexões residuais**
* **Layer Normalization**

```python
X1 = layer_norm(X + attn_output)
```

---

# 6. Feed Forward Network

Cada token passa por uma rede neural totalmente conectada.

```
FFN(x) = max(0, xW1 + b1)W2 + b2
```

Implementação:

```python
hidden = np.maximum(0, X @ W1 + b1)
output = hidden @ W2 + b2
```

---

# 7. Encoder Layer

Cada camada do encoder contém:

1. Self-Attention
2. Residual + LayerNorm
3. Feed Forward
4. Residual + LayerNorm

---

# 8. Empilhamento de Camadas

O Transformer utiliza várias camadas empilhadas.

Neste exemplo foram usadas **6 camadas**.

```python
for i in range(num_layers):
    output = encoder_layer(output)
```

---

# Execução

Para executar o código:

```bash
python transformer_encoder.py
```

Saída esperada:

```
Shape da saída: (1, 4, 64)
```

Onde:

* **1** = batch size
* **4** = número de tokens na frase
* **64** = dimensão do embedding

---


