import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Sample user-item interaction data
num_users = 1000
num_items = 500
user_ids = np.random.randint(1, num_users, size=10000)
item_ids = np.random.randint(1, num_items, size=10000)
ratings = np.random.randint(1, 6, size=10000)

# Define autoencoder model
input_layer = Input(shape=(2,))
encoded = Dense(32, activation='relu')(input_layer)
decoded = Dense(2, activation='sigmoid')(encoded)
autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

# Compile and train the model
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(np.column_stack((user_ids, item_ids)), np.column_stack((user_ids, item_ids)), epochs=10, batch_size=32)

# Generate recommendations for users
user_id = 123
items = np.arange(1, num_items + 1)
encoded_user = np.array([[user_id, item_id] for item_id in items])
encoded_items = encoder.predict(encoded_user)
predictions = autoencoder.predict(encoded_items)
top_items = items[np.argsort(predictions[:, 1])[::-1][:5]]
print("Recommendations for user", user_id, ":", top_items)