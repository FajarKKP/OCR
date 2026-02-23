from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    use_gpu=False,
    enable_mkldnn=False,
    show_log=False
)

result = ocr.ocr("test_img.jpg", cls=True)

for line in result[0]:
    text = line[1][0]
    conf = line[1][1]
    print(f"{text} ({conf:.3f})")