import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Dados de treinamento: Celsius (entrada) e Fahrenheit (saída)
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46.4, 59, 71.6, 100.4], dtype=float)

# Criando o modelo Keras
model = Sequential([
    Dense(units=1, input_shape=[1], activation='linear')  # Camada única
])

# Compilando o modelo
model.compile(optimizer=Adam(learning_rate=0.1), loss='mean_squared_error')

# Treinando o modelo
print("Treinando o modelo...")
history = model.fit(celsius, fahrenheit, epochs=500, verbose=0)

# Visualizando o erro após o treinamento
print("Treinamento finalizado!")
print("Erro final (Loss):", history.history['loss'][-1])

# Testando o modelo
test_input = np.array([100, -30, 25], dtype=float)
predictions = model.predict(test_input)

# Exibindo os resultados
for i, c in enumerate(test_input):
    print(f"{c}°C = {predictions[i][0]:.2f}°F")
