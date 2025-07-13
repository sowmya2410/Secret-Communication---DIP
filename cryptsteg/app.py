import os
import uuid
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from encryption import generate_key, encrypt_image, decrypt_image
from steganography_module import embed_data_lsb, extract_data_lsb, calculate_psnr, plot_histogram
from email.message import EmailMessage
import smtplib


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['HISTOGRAM_FOLDER'] = os.path.join('static', 'histograms')

# Ensure folders exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], app.config['HISTOGRAM_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/output/<filename>')
def serve_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


@app.route('/encrypt_embed', methods=['GET', 'POST'])
def encrypt_embed():
    if request.method == 'POST':
        password = request.form['password']
        auth_img = request.files['auth_img']
        secret_img = request.files['secret_img']
        secret_msg = request.form['secret_msg']
        email = request.form['email']

        # Validate file types
        ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
        def allowed_file(filename):
            return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

        if not (allowed_file(auth_img.filename) and allowed_file(secret_img.filename)):
            return "Only JPG, JPEG, and PNG files are allowed.", 400

        # Save uploaded files
        auth_img_filename = f"{uuid.uuid4()}_{auth_img.filename}"
        secret_img_filename = f"{uuid.uuid4()}_{secret_img.filename}"
        auth_img_path = os.path.join(app.config['UPLOAD_FOLDER'], auth_img_filename)
        secret_img_path = os.path.join(app.config['UPLOAD_FOLDER'], secret_img_filename)
        auth_img.save(auth_img_path)
        secret_img.save(secret_img_path)

        print(f"[INFO] Secret image path: {os.path.abspath(secret_img_path)}")

        # Check if image is valid
        if cv2.imread(secret_img_path) is None:
            raise ValueError(f"Error: Cannot load image at {secret_img_path}")

        # Encrypt
        key = generate_key(password, auth_img_path)
        encrypted_bytes, iv, shape = encrypt_image(secret_img_path, key)

        # Save encrypted binary
        encrypted_data_filename = f"{uuid.uuid4()}_encrypted.bin"
        encrypted_data_path = os.path.join(app.config['OUTPUT_FOLDER'], encrypted_data_filename)
        with open(encrypted_data_path, 'wb') as f:
            f.write(encrypted_bytes)

        # Save metadata
        metadata_filename = encrypted_data_filename.replace('.bin', '_meta.txt')
        metadata_path = os.path.join(app.config['OUTPUT_FOLDER'], metadata_filename)
        with open(metadata_path, 'w') as f:
            f.write(f"{iv.hex()}\n{shape[0]},{shape[1]},{shape[2]}")

        # Embed message
        stego_filename = f"{uuid.uuid4()}_stego.png"
        output_stego_path = os.path.join(app.config['OUTPUT_FOLDER'], stego_filename)
        _, _ = embed_data_lsb(secret_img_path, secret_msg.encode(), output_path=output_stego_path)
        
        
        
        # Send email with attachment
        send_email_with_attachments(
            to_email=email,
            subject="Your Encrypted Stego Image & Authentication Image",
            body="Attached are your stego image and authentication image used for encryption.",
            password=password,
            attachment_paths=[output_stego_path, auth_img_path]
        )



        download_url = url_for('serve_output', filename=stego_filename)

        return render_template('encrypt_embed.html', stego_image=stego_filename, download_url=download_url)

    return render_template('encrypt_embed.html', stego_image=None, download_url=None)


@app.route('/decrypt_extract', methods=['GET', 'POST'])
def decrypt_extract():
    if request.method == 'POST':
        password = request.form['password']
        auth_img = request.files['auth_img']
        stego_img = request.files['stego_img']

        # Save files
        auth_img_filename = f"{uuid.uuid4()}_{auth_img.filename}"
        stego_img_filename = f"{uuid.uuid4()}_{stego_img.filename}"
        auth_img_path = os.path.join(app.config['UPLOAD_FOLDER'], auth_img_filename)
        stego_img_path = os.path.join(app.config['UPLOAD_FOLDER'], stego_img_filename)
        auth_img.save(auth_img_path)
        stego_img.save(stego_img_path)

        key = generate_key(password, auth_img_path)

        extracted_data, shape = extract_data_lsb(stego_img_path)
        secret_msg = extracted_data.decode() if extracted_data else "Extraction failed."

        # Try to find the corresponding encrypted bin and meta file
        encrypted_bin_file = None
        meta_file = None
        for file in os.listdir(app.config['OUTPUT_FOLDER']):
            if file.endswith('.bin'):
                encrypted_bin_file = os.path.join(app.config['OUTPUT_FOLDER'], file)
                meta_candidate = file.replace('.bin', '_meta.txt')
                if meta_candidate in os.listdir(app.config['OUTPUT_FOLDER']):
                    meta_file = os.path.join(app.config['OUTPUT_FOLDER'], meta_candidate)
                    break

        if encrypted_bin_file and meta_file:
            with open(encrypted_bin_file, 'rb') as f:
                encrypted_data = f.read()
            with open(meta_file, 'r') as f:
                iv_hex = f.readline().strip()
                shape_str = f.readline().strip()
            iv = bytes.fromhex(iv_hex)
            shape = tuple(map(int, shape_str.split(',')))

            decrypted_img = decrypt_image(encrypted_data, key, iv, shape)

            # Save decrypted image
            decrypted_filename = f"{uuid.uuid4()}_decrypted.png"
            decrypted_path = os.path.join(app.config['OUTPUT_FOLDER'], decrypted_filename)
            cv2.imwrite(decrypted_path, decrypted_img)
        else:
            decrypted_filename = None

        # Save histogram
        stego_img_cv2 = cv2.imread(stego_img_path)
        histogram_path = os.path.join(app.config['HISTOGRAM_FOLDER'], f"{uuid.uuid4()}_histogram.png")
        plot_histogram(stego_img_cv2, histogram_path)

        return render_template(
            'decrypt_extract.html',
            secret_msg=secret_msg,
            decrypted_image=decrypted_filename,
            histogram_image=histogram_path
        )

    return render_template('decrypt_extract.html', secret_msg=None, decrypted_image=None, histogram_image=None)



@app.route('/download-stego')
def download_stego():
    return send_file('stego.png', as_attachment=True)


def send_email_with_attachments(to_email, subject, body, password, attachment_paths):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = 'kiruthigaji2004@gmail.com'
    msg['To'] = to_email
    msg.set_content(f"{body}\n\nPassword: {password}")

    for path in attachment_paths:
        with open(path, 'rb') as f:
            file_data = f.read()
            file_name = os.path.basename(path)
            maintype, subtype = 'application', 'octet-stream'
            if file_name.lower().endswith(('png', 'jpg', 'jpeg')):
                maintype, subtype = 'image', file_name.split('.')[-1]
            msg.add_attachment(file_data, maintype=maintype, subtype=subtype, filename=file_name)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login('kiruthigaji2004@gmail.com', 'mqgrzsbksfnkmrci')  # App password
        smtp.send_message(msg)



if __name__ == '__main__':
    app.run(debug=True)
