from fastapi import FastAPI, File, UploadFile, Response
import time
import cupy as cp
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()


@app.post("/histogram/")
async def histogram(file: UploadFile = File(...)):
    start = time.time()
    # 读取图像
    contents = await file.read()
    if contents is None:
        return Response(content="No image data", status_code=400)
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    # 将图像数据从CPU转移到GPU
    image_gpu = cp.asarray(image)
    # 计算直方图
    hist_gpu = cp.histogram(image_gpu, bins=256, range=(0, 256))[0]
    # 将直方图数据从GPU转移回CPU
    hist = cp.asnumpy(hist_gpu)
    # 归一化直方图，使其最大值为100
    hist = hist / hist.max() * 100
    # 创建一个100x256的黑色图像，以留出空间给X轴和Y轴标签
    hist_image = np.zeros((100, 256), dtype=np.uint8)

    for x in range(256):
        cv2.line(hist_image, (x, 100), (x, 100 - int(hist[x])), 255)

    hist_image = cv2.imencode('.png', hist_image)[1].tobytes()

    end = time.time()
    # 返回直方图
    return Response(content=hist_image, media_type="image/png", headers={"Time": str(end - start)})


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
