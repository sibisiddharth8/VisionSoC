from flask import Flask, request, render_template, send_file
import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
import base64
import io  
from flask import send_file

app = Flask(__name__)

model_path = 'RRDB_ESRGAN_x4.pth' 
device = torch.device('cpu')  
test_img_folder = 'LR/*'

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()

def process_image(file_stream):
    img = cv2.imdecode(np.frombuffer(file_stream.read(), np.uint8), cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    _, buffer = cv2.imencode('.jpg', output)
    return base64.b64encode(buffer).decode('utf-8'), output  # Returning both base64 encoded image and raw image data

@app.route('/', methods=['GET', 'POST'])
def index():
    img_data = None
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            img_data, output_image = process_image(uploaded_file)  # Modify to get the output image data
            # Save output image data to global variable for download
            global output_image_data
            _, buffer = cv2.imencode('.jpg', output_image)
            output_image_data = io.BytesIO(buffer)
    return render_template('index.html', img_data=img_data)

@app.route('/download')
def download():
    global output_image_data
    if output_image_data:
        output_image_data.seek(0)
        return send_file(output_image_data,
                         mimetype='image/jpeg',
                         as_attachment=True,
                         download_name='output_image.jpg')  
    else:
        return "Output image is not available for download."

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
