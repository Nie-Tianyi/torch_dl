import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt


def render_hanzi_to_gray(
    char, size=32, font_path="simsun.ttc"
):  # Windows通常是SimSun.ttf, Mac是Arial Unicode.ttf
    # 1. 创建画布 (使用 'L' 模式，背景为黑色 0)
    canvas_size = 64
    image = Image.new("L", (canvas_size, canvas_size), color=0)
    draw = ImageDraw.Draw(image)

    try:
        # 尝试加载字体，如果失败会进入 except
        font = ImageFont.truetype(font_path, int(canvas_size * 0.8))
    except OSError:
        print(f"警告：未找到字体 {font_path}，请检查系统字体库。")
        return None

    # 2. 精确计算文字中心
    # 使用 font.getbbox 替代 draw.textbbox 更为可靠
    left, top, right, bottom = font.getbbox(char)
    text_width = right - left
    text_height = bottom - top

    # 计算居中坐标 (考虑到某些字体自带偏移，需要减去 left 和 top)
    x = (canvas_size - text_width) // 2 - left
    y = (canvas_size - text_height) // 2 - top

    # 3. 渲染 (填充白色 255)
    draw.text((x, y), char, font=font, fill=255)

    # 4. 调试：如果还是全零，先看看 64x64 的原图是否有东西
    if np.sum(np.array(image)) == 0:
        print(f"错误：字符 '{char}' 渲染失败，画布仍为全黑。")
        return np.zeros((size, size))

    # 5. 缩小
    low_res_img = image.resize((size, size), Image.Resampling.LANCZOS)
    return np.array(low_res_img)


def display_gray_matrix_binary(matrix, threshold=64):
    # 显示矩阵 (二值化显示)
    binary_matrix = (matrix > threshold).astype(int)
    plt.imshow(binary_matrix, cmap="gray")
    plt.show()


def display_gray_matrix(matrix):
    plt.imshow(matrix, cmap="gray")
    plt.show()


def get_common_chars():
    # 获取GB2312常用汉字范围 (一级字库 3755个)
    chars = []
    for i in range(0xB0, 0xD7):
        for j in range(0xA1, 0xFF):
            try:
                char = bytes([i, j]).decode("gb2312")
                chars.append(char)
            except:
                continue
    return chars[:3500]


if __name__ == "__main__":
    # 快速测试
    matrix = render_hanzi_to_gray("abstract", size=32)
    display_gray_matrix(matrix)
