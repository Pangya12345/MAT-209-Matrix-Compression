"""Interactive demo: image compression with SVD (matrix compression).

วิธีรัน:   python demo.py

- ลากแถบเลื่อน (หรือกดลูกศร ซ้าย/ขวา = ทีละ 1, ขึ้น/ลง = ทีละ 10) เพื่อเลือก k
  คือจำนวน "ชั้น" ของ SVD ที่เราเก็บไว้
- คลิกชื่อสัตว์ด้านซ้ายล่างเพื่อสลับภาพ
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RadioButtons, Slider
from PIL import Image

# ---------------------------------------------------------------------------
# ตั้งค่า
# ---------------------------------------------------------------------------
IMAGE_DIR = Path(__file__).parent / "images"
IMAGES = {"Shark": "shark.jpg", "Deer": "deer.jpg", "Leopard": "leopard.jpg"}
K_MAX = 300    # ค่า k สูงสุดบนแถบเลื่อน
K_START = 20   # ค่า k ตอนเปิดโปรแกรม


# ---------------------------------------------------------------------------
# ส่วนคณิตศาสตร์ (หัวใจของโปรเจกต์)
# ---------------------------------------------------------------------------
def load(filename):
    """โหลดภาพ แล้วทำ SVD แยกทีละสี

    ภาพสี = เมทริกซ์ 3 อันซ้อนกัน (R, G, B) แต่ละอันขนาด m x n
    ค่าในแต่ละช่องคือความสว่างของพิกเซล 0-255
    """
    # แปลงภาพเป็นเมทริกซ์ตัวเลข shape = (m, n, 3)
    img = np.asarray(Image.open(IMAGE_DIR / filename).convert("RGB"), dtype=float)

    # ทำ SVD ทีละสี:  A = U  S  V^T
    #   U  = เวกเตอร์แนวตั้ง (m x r)
    #   S  = ค่าเอกฐาน σ1 >= σ2 >= ... บอกความสำคัญของแต่ละชั้น
    #   Vt = เวกเตอร์แนวนอน (r x n)
    # full_matrices=False -> ได้ SVD แบบย่อ ประหยัดหน่วยความจำ
    svds = [np.linalg.svd(img[:, :, c], full_matrices=False) for c in range(3)]

    # ทำ SVD ครั้งเดียวตอนเปิดโปรแกรม ตอนลากแถบเลื่อนแค่ตัดเอา k ชั้นแรก จึงเร็ว
    return img, svds


def compress(svds, k):
    """เก็บแค่ k ชั้นแรกของแต่ละสี:  A_k = U_k  S_k  V_k^T

    เท่ากับ  σ1·u1·v1^T + σ2·u2·v2^T + ... + σk·uk·vk^T
    ตามทฤษฎีบท Eckart–Young นี่คือการประมาณ rank-k ที่ดีที่สุดที่เป็นไปได้
    """
    channels = [
        U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]  # k คอลัมน์แรกของ U, k ค่าแรกของ S, k แถวแรกของ Vt
        for U, S, Vt in svds                   # ทำทีละสี R, G, B
    ]
    # รวม 3 สีกลับเป็นภาพ แล้วตัดค่าให้อยู่ในช่วง 0-255 (ค่าที่ประมาณอาจเกินขอบเล็กน้อย)
    return np.clip(np.stack(channels, axis=2), 0, 255).astype(np.uint8)


def storage(img, k):
    """สัดส่วนพื้นที่ที่ใช้เมื่อเทียบกับภาพต้นฉบับ

    ภาพเดิมเก็บ m*n ตัวเลขต่อสี
    แบบบีบอัดเก็บ U_k (m*k) + S_k (k) + V_k (n*k) = k(m + n + 1) ตัวเลขต่อสี
    """
    m, n, _ = img.shape
    return k * (m + n + 1) / (m * n)


def psnr(img, approx):
    """คุณภาพภาพ (หน่วย dB) ยิ่งสูงยิ่งเหมือนต้นฉบับ

    PSNR = 10 log10(255^2 / MSE) โดย MSE = ค่าเฉลี่ยของ (ความต่างของพิกเซล)^2
    """
    mse = np.mean((img - approx) ** 2)
    return 10 * np.log10(255 ** 2 / mse)


def quality_label(db):
    """แปลงค่า dB เป็นคำที่คนฟังเข้าใจง่าย"""
    if db < 22:
        return "poor"
    if db < 27:
        return "fair"
    if db < 32:
        return "good"
    if db < 40:
        return "very good"
    return "almost identical"


# ---------------------------------------------------------------------------
# ส่วนหน้าจอ demo
# ---------------------------------------------------------------------------
def build_demo():
    # ทำ SVD ของทุกภาพไว้ล่วงหน้า สลับภาพแล้วจะได้ไม่ต้องรอ
    print("Computing SVD for each image...")
    data = {name: load(filename) for name, filename in IMAGES.items()}
    state = {"name": "Shark"}  # ภาพที่กำลังแสดงอยู่

    # --- จัดวางหน้าจอ: [ซ้าย, ล่าง, กว้าง, สูง] เป็นสัดส่วนของหน้าต่าง ---
    fig = plt.figure(figsize=(14, 8.5))
    fig.suptitle("Matrix Compression with SVD", fontsize=20, fontweight="bold")

    ax_orig = fig.add_axes([0.03, 0.40, 0.45, 0.50])    # ภาพต้นฉบับ (ซ้ายบน)
    ax_comp = fig.add_axes([0.52, 0.40, 0.45, 0.50])    # ภาพที่บีบอัด (ขวาบน)
    ax_slider = fig.add_axes([0.20, 0.31, 0.62, 0.03])  # แถบเลื่อน k
    ax_radio = fig.add_axes([0.03, 0.05, 0.13, 0.20])   # ปุ่มเลือกภาพ (ซ้ายล่าง)
    ax_sv = fig.add_axes([0.26, 0.07, 0.71, 0.18])      # กราฟค่าเอกฐาน (ขวาล่าง)
    info = fig.text(0.5, 0.365, "", ha="center", fontsize=15)  # ข้อความ พื้นที่ + คุณภาพ

    # --- ภาพต้นฉบับกับภาพที่บีบอัด ---
    img, svds = data[state["name"]]
    orig_view = ax_orig.imshow(img.astype(np.uint8))
    comp_view = ax_comp.imshow(compress(svds, K_START))
    ax_orig.set_title("Original", fontsize=15)
    for ax in (ax_orig, ax_comp):
        ax.axis("off")  # ซ่อนแกนตัวเลขรอบภาพ

    # --- แถบเลื่อนกับปุ่มเลือกภาพ ---
    slider = Slider(ax_slider, "k (layers kept)", 1, K_MAX, valinit=K_START, valstep=1)
    slider.vline.set_visible(False)  # ซ่อนขีดแดงที่บอกค่าเริ่มต้น
    radio = RadioButtons(ax_radio, list(IMAGES))
    ax_radio.set_title("Image", fontsize=13)

    # --- กราฟค่าเอกฐาน σi: ชี้ให้เห็นว่าชั้นแรก ๆ สำคัญกว่าชั้นหลังมาก ---
    ax_sv.set_yscale("log")  # สเกล log เพราะ σ1 ใหญ่กว่าตัวท้าย ๆ หลายร้อยเท่า
    ax_sv.set_xlim(1, K_MAX)
    ax_sv.set_xlabel("layer i")
    ax_sv.set_ylabel("importance σᵢ")
    ax_sv.set_title("Each layer is less important than the one before", fontsize=12)
    sv_line, = ax_sv.plot([], [], color="#1565c0", lw=2)          # เส้นกราฟ σi
    cut_line = ax_sv.axvline(K_START, color="#c62828", lw=2)       # เส้นแดง = จุดที่ตัด (k)
    kept_area = [None]                                             # พื้นที่แรเงา = ชั้นที่เก็บไว้

    def redraw(_=None):
        """คำนวณและวาดใหม่ทุกครั้งที่ k หรือภาพเปลี่ยน"""
        img, svds = data[state["name"]]
        k = int(slider.val)

        # 1) บีบอัดด้วย k ชั้นแล้วแสดงผล
        approx = compress(svds, k)
        comp_view.set_data(approx)
        ax_comp.set_title(f"Compressed (k = {k})", fontsize=15)

        # 2) อัปเดตตัวเลข พื้นที่ที่ใช้ + คุณภาพ
        db = psnr(img, approx)
        info.set_text(f"Storage used: {storage(img, k):.1%} of the original    ·    "
                      f"Quality: {db:.1f} dB ({quality_label(db)})")

        # 3) อัปเดตกราฟ σi (เฉลี่ยของ 3 สี) แล้วแรเงาส่วนที่เก็บไว้
        S = np.mean([s for _, s, _ in svds], axis=0)
        i = np.arange(1, len(S) + 1)
        floor = S[K_MAX - 1] * 0.8  # ขอบล่างของกราฟ
        sv_line.set_data(i, S)
        ax_sv.set_ylim(floor, S[0] * 1.5)
        cut_line.set_xdata([k, k])
        if kept_area[0] is not None:
            kept_area[0].remove()  # ลบแรเงาเก่าก่อนวาดใหม่
        kept_area[0] = ax_sv.fill_between(i[:k], S[:k], floor, color="#1565c0", alpha=0.2)

        fig.canvas.draw_idle()  # สั่งให้หน้าจอวาดใหม่

    def change_image(name):
        """เมื่อคลิกเลือกภาพใหม่"""
        state["name"] = name
        img = data[name][0]
        m, n, _ = img.shape
        orig_view.set_data(img.astype(np.uint8))
        # ภาพแต่ละภาพสูงไม่เท่ากัน ต้องปรับขนาดกรอบให้ตรงกับภาพใหม่
        for view, ax in ((orig_view, ax_orig), (comp_view, ax_comp)):
            view.set_extent((-0.5, n - 0.5, m - 0.5, -0.5))
            ax.set_xlim(-0.5, n - 0.5)
            ax.set_ylim(m - 0.5, -0.5)
        redraw()

    def on_key(event):
        """ปุ่มลูกศร: ซ้าย/ขวา เปลี่ยน k ทีละ 1, ขึ้น/ลง ทีละ 10 (สะดวกตอนพรีเซนต์)"""
        step = {"right": 1, "left": -1, "up": 10, "down": -10}.get(event.key)
        if step:
            slider.set_val(min(max(slider.val + step, 1), K_MAX))

    # ผูกเหตุการณ์: ลากแถบเลื่อน -> redraw, คลิกเลือกภาพ -> change_image, กดปุ่ม -> on_key
    slider.on_changed(redraw)
    radio.on_clicked(change_image)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()  # วาดครั้งแรก
    return fig, (slider, radio)


if __name__ == "__main__":
    fig, widgets = build_demo()  # เก็บ widgets ไว้ในตัวแปร ไม่งั้นแถบเลื่อนจะหยุดทำงาน
    plt.show()
