from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import mnist
from keras.utils import to_categorical

# Carregar o dataset MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar os valores dos pixels (0 a 255 para 0 a 1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode das labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Criar o modelo sequencial
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Converte imagens 2D para 1D
    Dense(128, activation='relu'),  # Camada densa com 128 neurônios
    Dense(64, activation='relu'),   # Camada densa com 64 neurônios
    Dense(10, activation='softmax') # Saída com 10 classes
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Avaliar o modelo no conjunto de teste
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
