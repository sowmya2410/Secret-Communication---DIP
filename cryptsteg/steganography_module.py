import numpy as np
import cv2
import hashlib
import gc
import zlib
import matplotlib
matplotlib.use('Agg')  # Use Anti-Grain Geometry backend (no GUI)
import matplotlib.pyplot as plt



def str_to_bits(data_bytes):
    return ''.join(f'{byte:08b}' for byte in data_bytes)

def bits_to_bytes(bit_str):
    return bytes(int(bit_str[i:i+8], 2) for i in range(0, len(bit_str), 8))

def hash_bytes(data_bytes):
    return hashlib.sha256(data_bytes).digest()

def compress_payload(data_bytes, tag=b'TXT'):
    compressed = zlib.compress(data_bytes)
    payload = tag + len(compressed).to_bytes(4, 'big') + hash_bytes(compressed) + compressed
    return payload

def decompress_payload(payload):
    tag = payload[:3]
    length = int.from_bytes(payload[3:7], 'big')
    hash_check = payload[7:39]
    compressed_data = payload[39:39+length]
    assert hash_bytes(compressed_data) == hash_check, "Hash mismatch: Data integrity check failed"
    return zlib.decompress(compressed_data), tag.decode()

def embed_data_lsb(image_path, data_bytes, output_path='stego.png', lsb_count=1):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Image not found or unreadable")

    if len(img.shape) == 2:  # grayscale
        img = np.expand_dims(img, axis=-1)

    flat_img = img.reshape(-1, img.shape[2])
    total_bytes = flat_img.size * lsb_count // 8

    payload = compress_payload(data_bytes)
    bits = str_to_bits(payload)
    if len(bits) > flat_img.size * lsb_count:
        raise ValueError("Payload too large for image capacity")

    bit_index = 0
    for pixel in flat_img:
        for c in range(img.shape[2]):
            for b in range(lsb_count):
                if bit_index < len(bits):
                    mask = 255 - (1 << b)  # Always stays within 0-255, same as bitwise AND mask

                    bit_val = np.uint8(int(bits[bit_index]) << b)
                    pixel[c] = (pixel[c] & mask) | bit_val
                    bit_index += 1

                    

    stego = flat_img.reshape(img.shape)
    cv2.imwrite(output_path, stego)
    print(f"[✔] Embedded data into {output_path} ({len(bits)} bits)")
    return img, stego

def extract_data_lsb(stego_path, lsb_count=1):
    img = cv2.imread(stego_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Image not found or unreadable")

    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)

    flat_img = img.reshape(-1, img.shape[2])
    bits = ''
    for pixel in flat_img:
        for c in range(img.shape[2]):
            for b in range(lsb_count):
                bits += str((pixel[c] >> b) & 1)

    byte_data = bits_to_bytes(bits)
    try:
        data, tag = decompress_payload(byte_data)
        print(f"[✔] Extracted {len(data)} bytes of {tag} data successfully")
        return data, tag
    except Exception as e:
        print(f"[✖] Extraction failed: {e}")
        return None, None

def calculate_psnr(original, stego):
    original = original.astype(np.float32)
    stego = stego.astype(np.float32)
    mse = np.mean((original - stego) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

"""def plot_histograms(original, stego):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].set_title("Original Image Histogram")
    axs[1].set_title("Stego Image Histogram")

    for i, title in zip(range(original.shape[2]), ['R', 'G', 'B'] if original.shape[2] == 3 else ['Gray']):
        axs[0].hist(original[:, :, i].ravel(), bins=256, color=title.lower(), alpha=0.6)
        axs[1].hist(stego[:, :, i].ravel(), bins=256, color=title.lower(), alpha=0.6)

    plt.tight_layout()
    plt.show()
"""


def plot_histogram(image, save_path="histogram.png"):
    import matplotlib.pyplot as plt
    import gc

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_title("Stego Image Histogram")

    channels = ['R', 'G', 'B'] if image.shape[2] == 3 else ['Gray']
    for i, title in enumerate(channels):
        ax.hist(image[:, :, i].ravel(), bins=256, color=title.lower(), alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close('all')
    gc.collect()
    # Clear unused memory references

