from diffusers import StableDiffusionPipeline

# Load pre-trained model from the Hugging Face hub
model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
model.to("cuda")  # Move the model to GPU for faster processing
def generate_image(prompt):
    image = model(prompt).images[0]
    image.save("output.png")
    return "output.png"
from flask import Flask, request, send_file

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']
    image_path = generate_image(prompt)
    return send_file(image_path, mimetype='image/png')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
