from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError
import io
import base64
import logging

from model import predict_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# 메인 페이지 렌더링
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# 이미지 업로드 및 예측 처리
@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # 이미지 예측
        prediction = predict_image(image)

        # 이미지 base64 인코딩 (결과 페이지에 표시용)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": prediction,
            "image_data": img_str
        })

    except UnidentifiedImageError:
        logging.warning("업로드된 파일이 이미지가 아닙니다.")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": {"class": "유효한 이미지 파일이 아닙니다.", "confidence": 0, "index": -1}
        })
    except Exception as e:
        logging.error("예측 중 오류 발생: " + str(e))
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": {"class": f"서버 오류: {str(e)}", "confidence": 0, "index": -1}
        })
