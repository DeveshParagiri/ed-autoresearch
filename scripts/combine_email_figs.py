from PIL import Image

m = Image.open(r"D:\FIRE_OFFLINE\NEW MAPS\continental_model\01_burned_area_map.png").convert("RGB")
s = Image.open(r"D:\FIRE_OFFLINE\NEW MAPS\continental_model\05_per_cell_scatter.png").convert("RGB")

W = m.width  # 2646
sc_w = 1180
sc_h = round(s.height * sc_w / s.width)
s = s.resize((sc_w, sc_h), Image.LANCZOS)

gap = 50
H = m.height + gap + sc_h
canvas = Image.new("RGB", (W, H), "white")
canvas.paste(m, (0, 0))
canvas.paste(s, ((W - sc_w) // 2, m.height + gap))

out = r"D:\FIRE_OFFLINE\NEW MAPS\continental_model\for_email_ba_map_plus_scatter.png"
canvas.save(out, dpi=(200, 200))
print("saved", out, canvas.size)
