from typing import Union
from fastapi import FastAPI, UploadFile, File
import fitz  
import optik_reader_yolo

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/read_pdf")
async def read_pdf(file: UploadFile = File(...)):
    with fitz.open(stream=file.file.read(), filetype="pdf") as doc:
        if doc.page_count > 0:
                first_page = doc.load_page(0)
                pix = first_page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72)) 
                filename = "images/temp.png"
                pix.save(filename)
    result = optik_reader_yolo.read_optik_form()
    return result

@app.post("/read_image")
async def read_image(file: UploadFile = File(...)):
    with open("images/temp.png", "wb") as buffer:
        buffer.write(file.file.read())
    result = optik_reader_yolo.read_optik_form()
    return result