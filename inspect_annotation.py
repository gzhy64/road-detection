"""
    读取一张路标图片的 XML 标注，打印目标信息，
    并在原图上绘制真实标注框，检查图片与标注是否对应.
"""


from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw


project_dir = Path(__file__).resolve().parent

annotation_path = project_dir / "archive" / "annotations" / "road0.xml"

# 解析 XML，取得最外层的 annotation 节点。
root = ET.parse(annotation_path).getroot()

# XML 中读取的数字是字符串，需要转换成整数。
# 图片尺寸的顺序为：宽度、高度，单位均为像素。
filename = root.findtext("filename")
width = int(root.findtext("size/width"))
height = int(root.findtext("size/height"))

# 每个字典保存一个目标的类别和边框；一张图可以包含多个目标。
objects = []

for obj in root.findall("object"):
    label = obj.findtext("name")
    box = obj.find("bndbox")

    coordinates = [
        int(box.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax")
    ]

    objects.append({
        "label": label,
        "box": coordinates,
    })


print("图片文件名：", filename)
print("图片宽、高：", width, height)
print("目标数量：", len(objects))

for item in objects:
    print("类别：", item["label"], "边框：", item["box"])


# 使用 XML 中的文件名寻找原图，并统一为 RGB 三通道格式
image_path = project_dir / "archive" / "images"  / filename

with Image.open(image_path) as source:
    image = source.convert("RGB")

# 创建绘图对象，在内存中的图片上绘制真实标注
draw = ImageDraw.Draw(image)

for item in objects:
    xmin, ymin, xmax, ymax = item["box"]

    draw.rectangle(
        [xmin, ymin, xmax, ymax],
        outline="red",
        width=3,
    )
    # 将类别文字放在框上方，文字纵坐标至少为 0
    draw.text(
        (xmin, max(0, ymin - 15)),
        item["label"],
        fill="yellow",
    )

# 将预览图保存到独立的输出目录
# gt 表示 ground truth，即数据集提供的真实标注
output_dir = project_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / f"{Path(filename).stem}_gt.png"
image.save(output_path)

print("实际图片宽、高：", image.size)
print("已保存标注预览：", output_path)
