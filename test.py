import numpy as np

# Function to compute cosine similarity
def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# Loss calculation function
def calculate_loss(alpha_embeddings, beta_embeddings):
    N = alpha_embeddings.shape[0]
    loss = 0
    for i in range(N):
        cos_ai_bi = cosine_similarity(alpha_embeddings[i], beta_embeddings[i])
        
        sum_cos_ai_bj = 0
        for j in range(N):
            if i != j:
                sum_cos_ai_bj += cosine_similarity(alpha_embeddings[i], beta_embeddings[j])
        
        loss += -cos_ai_bi / sum_cos_ai_bj
    return loss

# Initialize base embeddings
N, D = 3, 2  # Batch size and dimensionality
base_embeddings = np.random.randn(N, D)

# Define alpha and beta coefficients
alpha_coefficients = np.array([1, 2])  # Example transformation coefficients for alpha
beta_coefficients = np.array([1, 0.5])  # Example transformation coefficients for beta

# Calculate and print the loss
loss = calculate_loss(alpha_coefficients, beta_coefficients)
print(f"Loss: {loss}")

# Calculate and print the loss
loss = calculate_loss(beta_coefficients, alpha_coefficients)
print(f"Loss: {loss}")
