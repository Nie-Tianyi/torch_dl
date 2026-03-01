from PIL import Image

# 1. 打开你的原图 (请确保路径正确)
img_path = "target.png"  # 如果不在同级目录，请写绝对路径
img = Image.open(img_path)

# 2. 修改尺寸：PIL 的 resize 接收的参数顺序是 (宽, 高)
new_size = (2560, 1600)
resized_img = img.resize(new_size, Image.Resampling.LANCZOS) # 使用高质量重采样

# 3. 保存为新图片或覆盖原图
resized_img.save("target_2560x1600.png")
print("图片尺寸修改成功！")