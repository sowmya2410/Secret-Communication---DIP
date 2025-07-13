import cv2
import numpy as np
import hashlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes


# ------------------ Utility Functions ------------------ #

def generate_key(password: str, auth_img_path: str) -> bytes:
    """
    Combine password and hash of an auth image to generate a strong 16-byte AES key.
    """
    auth_img = cv2.imread(auth_img_path, cv2.IMREAD_GRAYSCALE)
    auth_img_bytes = auth_img.tobytes()
    combined = password.encode() + auth_img_bytes
    key = hashlib.sha256(combined).digest()[:16]  # AES-128 key
    return key

def compress_image(secret_img_path: str, scale_percent: int = 50) -> np.ndarray:
    """
    Compress the image by resizing it to a given scale percentage.
    """
    # Read the image
    secret_img = cv2.imread(secret_img_path)
    
    # Check if image is loaded properly
    if secret_img is None:
        raise ValueError(f"Error: The image at {secret_img_path} could not be loaded. Please check the file path.")

    # Get current dimensions
    width = int(secret_img.shape[1] * scale_percent / 100)
    height = int(secret_img.shape[0] * scale_percent / 100)
    
    # Resize image (compression)
    compressed_img = cv2.resize(secret_img, (width, height), interpolation=cv2.INTER_AREA)
    
    return compressed_img

def encrypt_image(secret_img_path: str, key: bytes, scale_percent: int = 50) -> tuple:
    """
    Encrypt the secret image after compressing it using AES-CBC mode.
    Returns encrypted byte stream and IV.
    """
    # Compress the image before encryption
    secret_img = compress_image(secret_img_path, scale_percent)
    
    # Convert the image to bytes (flattened)
    secret_bytes = secret_img.tobytes()

    # Generate a random initialization vector (IV)
    iv = get_random_bytes(16)

    # Create AES cipher using CBC mode
    cipher = AES.new(key, AES.MODE_CBC, iv)
    
    # Encrypt the byte data, with padding to ensure it fits block size
    encrypted = cipher.encrypt(pad(secret_bytes, AES.block_size))

    return encrypted, iv, secret_img.shape

def decrypt_image(encrypted_data: bytes, key: bytes, iv: bytes, shape: tuple) -> np.ndarray:
    """
    Decrypt encrypted data and reshape it to the original image dimensions.
    """
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted = unpad(cipher.decrypt(encrypted_data), AES.block_size)
    
    # Reshape the decrypted data back to the original image shape
    img_array = np.frombuffer(decrypted, dtype=np.uint8).reshape(shape)
    
    return img_array


# ------------------ Main Code ------------------ #
"""
# Inputs
password = "cherry24"
auth_image = "rose.jpg"
secret_image = "profile.jpg"

# Generate AES key using password + auth image
key = generate_key(password, auth_image)

# Encrypt the secret image after compression
encrypted_bytes, iv, shape = encrypt_image(secret_image, key, scale_percent=50)  # 50% compression

# Now, the encrypted_bytes are returned instead of being written to a file
print(f"Encrypted bytes (size: {len(encrypted_bytes)} bytes) generated.")

# Decrypt the encrypted image
decrypted_img = decrypt_image(encrypted_bytes, key, iv, shape)

# Save the decrypted image to check the result
cv2.imwrite("decrypted_output.png", decrypted_img)
"""