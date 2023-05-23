from transformers import YolosImageProcessor, YolosForObjectDetection, pipeline
import gradio as gr
import requests
from PIL import Image, ImageDraw
import torch

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")


def predict(x):
    obj_detector = pipeline("object-detection", model=model,
                            feature_extractor=image_processor)
    return (str(obj_detector(x)), x)


with torch.no_grad():
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(
        outputs, threshold=0.5, target_sizes=target_sizes)[0]


draw = ImageDraw.Draw(image)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    x, y, x2, y2 = tuple(box)
    draw.rectangle((x, y, x2, y2), outline="red", width=1)
    draw.text((x, y), model.config.id2label[label.item()], fill="white")

# image.save("test.png")


with gr.Blocks() as demo:
    gr.Markdown("# Detect object from a given image ! ")
    image = gr.Image(label="Input Image", type="pil")
    output_text = gr.Textbox(label="Output Text")
    output_image = gr.Image(label="Output Image", type="pil")
    generate_btn = gr.Button("Generate")
    generate_btn.click(fn=predict, inputs=image, outputs=[
                       output_text, output_image])
    gr.Markdown("## Examples")
    gr.Examples(examples=["example.jpg"],
                cache_examples=True,
                inputs=image,
                outputs=[output_text, output_image],
                fn=predict)


demo.launch(server_name="0.0.0.0")
