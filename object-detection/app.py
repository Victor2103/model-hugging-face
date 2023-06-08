from transformers import YolosImageProcessor, YolosForObjectDetection, DetrImageProcessor, DetrForObjectDetection
import gradio as gr
from fastapi import FastAPI, UploadFile
from fastapi.responses import Response
from PIL import ImageDraw, Image
import torch
import io

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

app = FastAPI()

models = ['hustvl/yolos-small', 'hustvl/yolos-tiny', 'facebook/detr-resnet-50']


def predict(im, option_model):
    if option_model == 'facebook/detr-resnet-50':
        image_processor = DetrImageProcessor.from_pretrained(option_model)
        model = DetrForObjectDetection.from_pretrained(option_model)
    else:
        model = YolosForObjectDetection.from_pretrained(option_model)
        image_processor = YolosImageProcessor.from_pretrained(option_model)
    with torch.no_grad():
        inputs = image_processor(images=im, return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.tensor([im.size[::-1]])
        results = image_processor.post_process_object_detection(
            outputs, threshold=0.5, target_sizes=target_sizes)[0]
    draw = ImageDraw.Draw(im)
    text = ""
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        x, y, x2, y2 = tuple(box)
        draw.rectangle((x, y, x2, y2), outline="red", width=1)
        draw.text((x, y), model.config.id2label[label.item()], fill="white")
        text += f"Detected {model.config.id2label[label.item()]} with confidence "
        text += f"{round(score.item(), 3)} at location {box} \n"
    return ([text, im])


@app.post("/yolos-small")
async def create_upload_file(file: UploadFile):
    trans = file.file.read()
    trans = Image.open(io.BytesIO(trans))
    image = predict(trans, 'hustvl/yolos-small')[1]
    buf = io.BytesIO()
    image.save(buf, format='png')
    byte_im = buf.getvalue()
    return Response(content=byte_im, media_type="image/png")


@app.post("/yolos-tiny")
async def create_upload_file(file: UploadFile):
    trans = file.file.read()
    trans = Image.open(io.BytesIO(trans))
    print(type(trans))
    image = predict(trans, 'hustvl/yolos-tiny')[1]
    buf = io.BytesIO()
    image.save(buf, format='png')
    byte_im = buf.getvalue()
    return Response(content=byte_im, media_type="image/png")


@app.post("/detr-resnet-50")
async def create_upload_file(file: UploadFile):
    trans = file.file.read()
    trans = Image.open(io.BytesIO(trans))
    print(type(trans))
    image = predict(trans, 'facebook/detr-resnet-50')[1]
    buf = io.BytesIO()
    image.save(buf, format='png')
    byte_im = buf.getvalue()
    return Response(content=byte_im, media_type="image/png")


with gr.Blocks(title="Object-detection") as demo:
    gr.Markdown("# Detect object from a given image ! ")
    with gr.Column():
        options = gr.Dropdown(
            choices=models, label='Select Object Detection Model', show_label=True)
    with gr.Row():
        image = gr.Image(label="Input Image", type="pil")
        output_text = gr.Textbox(label="Output Text")
        output_image = gr.Image(label="Output Image", type="pil")
    generate_btn = gr.Button("Generate")
    generate_btn.click(fn=predict, inputs=[image, options], outputs=[
                       output_text, output_image])
    gr.Markdown("## Examples")
    gr.Examples(examples=[["examples/example_1.jpg", 'hustvl/yolos-tiny'],
                          ["examples/example_2.jpg", 'hustvl/yolos-small'],
                          ["examples/example_3.jpg", 'facebook/detr-resnet-50']],
                cache_examples=True,
                inputs=[image, options],
                outputs=[output_text, output_image],
                fn=predict)

app = gr.mount_gradio_app(app, demo, path='/')
