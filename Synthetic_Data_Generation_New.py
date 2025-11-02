# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Step 1: Load dataset
df = pd.read_csv("C:\\Users\\gargi\\Downloads\\Gargi_real_test_sample.csv")  # replace with your file name

# Step 2: Encode categorical columns if needed
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Step 3: Separate features and target
X = df.drop("target", axis=1)  # replace 'target' with your label column name
y = df["target"]

# Step 4: SMOTE to balance data
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Step 5: Normalize the data
scaler = MinMaxScaler()
X_res_scaled = scaler.fit_transform(X_res)

# Step 6: Define GAN components

latent_dim = 32
n_features = X_res_scaled.shape[1]

def build_generator():
    model = Sequential([
        Dense(64, input_dim=latent_dim),
        LeakyReLU(0.2),
        BatchNormalization(),
        Dense(128),
        LeakyReLU(0.2),
        BatchNormalization(),
        Dense(n_features, activation='tanh')
    ])
    return model

def build_discriminator():
    model = Sequential([
        Dense(128, input_dim=n_features),
        LeakyReLU(0.2),
        Dense(64),
        LeakyReLU(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
    return model

# Step 7: Build and compile GAN
generator = build_generator()
discriminator = build_discriminator()

discriminator.trainable = False
gan = Sequential([generator, discriminator])
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Step 8: Training loop
epochs = 5000
batch_size = 64
half_batch = batch_size // 2

d_losses = []
g_losses = []

for epoch in range(epochs):
    # Train discriminator
    idx = np.random.randint(0, X_res_scaled.shape[0], half_batch)
    real_samples = X_res_scaled[idx]
    noise = np.random.normal(0, 1, (half_batch, latent_dim))
    fake_samples = generator.predict(noise, verbose=0)

    d_loss_real = discriminator.train_on_batch(real_samples, np.ones((half_batch, 1)))
    d_loss_fake = discriminator.train_on_batch(fake_samples, np.zeros((half_batch, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    d_losses.append(d_loss[0])

    # Train generator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
    g_losses.append(g_loss)

    # Display progress
    if epoch % 500 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch} | D Loss: {d_loss[0]:.4f}, D Acc: {d_loss[1]*100:.2f}% | G Loss: {g_loss:.4f}")

# Step 9: Generate synthetic samples
noise = np.random.normal(0, 1, (1000, latent_dim))
synthetic_data = generator.predict(noise)

# Step 10: Inverse scaling if needed
synthetic_data_original = scaler.inverse_transform(synthetic_data)

# Step 11: Save synthetic data
synthetic_df = pd.DataFrame(synthetic_data_original, columns=X.columns)
synthetic_df.to_csv("synthetic_data.csv", index=False)
print("Synthetic data saved as 'synthetic_data.csv'")

# Step 12: Plot losses
plt.figure(figsize=(12, 6))
plt.plot(g_losses, label="Generator Loss")
plt.plot(d_losses, label="Discriminator Loss")
plt.title("GAN Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
