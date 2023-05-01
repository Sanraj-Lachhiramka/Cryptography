# Import necessary libraries
import numpy as np     # For mathematical operations on arrays
import wave            # For reading and writing audio files
import matplotlib.pyplot as plt   # For plotting signals

# Define function to generate a random orthogonal matrix of size n x n
def generate_key(n):
    """
    Generates a random orthogonal matrix of size n x n.
    """
    # Generate a random n x n matrix
    A = np.random.rand(n, n)
    # Compute the inverse of A
    A_inv = np.linalg.inv(A)
    # Perform QR decomposition on the inverse of A to get the orthogonal matrix Q
    Q, R = np.linalg.qr(A_inv)
    return Q


# Define function to wrap the given key by rotating it by 90 degrees and projecting it about the origin
def wrap_key(key):
    """
    Wraps the given key by rotating it by 90 degrees and projecting it about the origin.
    """
    # Rotate the key by 90 degrees and flip it horizontally
    return np.flip(np.transpose(key), axis=1)

# Define function to encrypt a signal using the given key (orthogonal matrix)
def encrypt(signal, key):
    """
    Encrypts the signal using the given key (orthogonal matrix).
    """
    # Determine the size of the key and the signal
    n = key.shape[0]
    signal_len = signal.shape[0]
    # Determine the amount of padding needed to make the signal a multiple of n
    pad_len = n - signal_len % n
    # Generate some random noise to pad the signal with
    noise_signal = np.random.RandomState(42).randn(pad_len) / 1000.0
    # Pad the signal with the noise
    padded_signal = np.concatenate((signal, noise_signal))
    # Reshape the padded signal into blocks of size n x 1
    blocks = np.reshape(padded_signal, (-1, n))
    # Encrypt each block by multiplying it by the key
    encrypted_signal = np.dot(blocks, key)
    # Flatten the encrypted signal and trim off any excess padding
    return encrypted_signal.flatten()[:signal_len]

# Define function to decrypt an encrypted signal using the given key (orthogonal matrix)
def decrypt(encrypted_signal, key):
    """
    Decrypts the encrypted signal using the given key (orthogonal matrix).
    """
    # Determine the size of the key and the encrypted signal
    n = key.shape[0]
    encrypted_signal_len = encrypted_signal.shape[0]
    # Determine the amount of padding needed to make the encrypted signal a multiple of n
    pad_len = n - encrypted_signal_len % n
    # Pad the encrypted signal with zeros
    padded_encrypted_signal = np.concatenate((encrypted_signal, np.zeros(pad_len)))
    # Reshape the padded encrypted signal into blocks of size n x 1
    blocks = np.reshape(padded_encrypted_signal, (-1, n))
    # Decrypt each block by multiplying it by the transpose of the key
    decrypted_signal = np.dot(blocks, key.T)
    # Flatten the decrypted signal and trim off any excess padding
    return decrypted_signal.flatten()[:encrypted_signal_len]

# Open audio file
audio_file = wave.open("C:/Users/kotag/Downloads/male.wav", "rb")

# Get audio parameters
sample_rate = audio_file.getframerate()
num_channels = audio_file.getnchannels()
sample_width = audio_file.getsampwidth()
num_frames = audio_file.getnframes()

# Read audio data
audio_data = audio_file.readframes(num_frames)

# Close audio file
audio_file.close()
# Convert audio data to numpy array
audio_signal = np.frombuffer(audio_data, dtype=np.int16)

# If stereo, reshape to separate channels
if num_channels == 2:
    audio_signal = np.reshape(audio_signal, (-1, 2))

# Normalize audio signal to range [-1, 1]
audio_signal = audio_signal / np.iinfo(np.int16).max

# Generate encryption key
key = generate_key(5)

# Wrap the key
wrapped_key = wrap_key(key)

# Encrypt the audio signal using the original key and the wrapped key
encrypted_signal = encrypt(audio_signal.flatten(), key)
wrapped_encrypted_signal = encrypt(audio_signal.flatten(), wrapped_key)

# Open a new wave file for writing the wrapped encrypted signal
output_file = wave.open("wrapped_encrypted_audio.wav", "wb")
output_file.setframerate(sample_rate)
output_file.setnchannels(num_channels)
output_file.setsampwidth(sample_width)

# Scale wrapped encrypted signal to the range of a 16-bit integer
wrapped_encrypted_signal = np.round(wrapped_encrypted_signal * np.iinfo(np.int16).max).astype(np.int16)

# Write the wrapped encrypted signal to the output file
output_file.writeframes(wrapped_encrypted_signal.flatten().tobytes())

# Close the output file
output_file.close()

# Decrypt the encrypted signal using the original key
decrypted_signal = decrypt(encrypted_signal, key)

# Reshape decrypted signal to original shape if stereo
if num_channels == 2:
    decrypted_signal = np.reshape(decrypted_signal, (-1, 2))

# Scale decrypted signal back to the original range
decrypted_signal = np.round(decrypted_signal * np.iinfo(np.int16).max).astype(np.int16)

# Open a new wave file for writing the decrypted signal
output_file = wave.open("decrypted_audio.wav", "wb")
output_file.setframerate(sample_rate)
output_file.setnchannels(num_channels)
output_file.setsampwidth(sample_width)

# Write the decrypted signal to the output file
output_file.writeframes(decrypted_signal.flatten().tobytes())

# Close the output file
output_file.close()

# Plot the original signal, encrypted signal, decrypted signal, and wrapped encrypted signal
time_axis = np.linspace(0, len(audio_signal) / sample_rate, len(audio_signal))
fig, axs = plt.subplots(4, 1, figsize=(8, 10))

axs[0].plot(time_axis, audio_signal)
axs[0].set_title("Original Signal")

axs[1].plot(time_axis, encrypted_signal)
axs[1].set_title("Encrypted Signal")

axs[2].plot(time_axis, decrypted_signal)
axs[2].set_title("Decrypted Signal")

axs[3].plot(time_axis, wrapped_encrypted_signal)
axs[3].set_title("Wrapped Encrypted Signal")

# Add titles and labels to the plot
plt.suptitle("Audio Encryption and Decryption")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

#Plot the encryption key and the wrapped key
fig, axs = plt.subplots(1, 2, figsize=(8, 4))

axs[0].imshow(key, cmap='gray')
axs[0].set_title("Encryption Key")

axs[1].imshow(wrapped_key, cmap='gray')
axs[1].set_title("Wrapped Key")

plt.tight_layout()
plt.show()