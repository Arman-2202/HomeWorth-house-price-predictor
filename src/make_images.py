"""
Draws every illustration used by the app (hero banner, city cards, property
types, logo) with Pillow.  No internet or stock photos are needed, so the
images are original and always available offline.

Run:  python -m src.make_images
"""
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import config as C

S = 2                       # super-sampling factor (draw big, shrink = smooth edges)
HOLE = (0, 0, 0, 0)         # fill colour that punches a transparent hole in a layer


# ------------------------------------------------------------------ colour utils
def hx(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def mix(a, b, t):
    return tuple(int(a[i] * (1 - t) + b[i] * t) for i in range(3))


def rgba(c, a=255):
    return (c[0], c[1], c[2], int(a))


def vgradient(w, h, stops):
    """stops: [(pos 0..1, (r,g,b)), ...]"""
    ys = np.linspace(0, 1, h)
    pos = [p for p, _ in stops]
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    for ch in range(3):
        col = np.interp(ys, pos, [c[ch] for _, c in stops])
        arr[:, :, ch] = col[:, None]
    arr[:, :, 3] = 255
    return Image.fromarray(arr, "RGBA")


def glow_layer(size, cx, cy, radius, color, strength=0.6):
    w, h = size
    yy, xx = np.ogrid[:h, :w]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius
    a = np.clip(1 - d, 0, 1) ** 2 * strength * 255
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[:, :, 0], arr[:, :, 1], arr[:, :, 2] = color
    arr[:, :, 3] = a.astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def new_layer(size):
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def over(base, layer, opacity=1.0):
    if opacity < 1.0:
        layer = layer.copy()
        a = layer.getchannel("A").point(lambda v: int(v * opacity))
        layer.putalpha(a)
    return Image.alpha_composite(base, layer)


# ------------------------------------------------------------------ building blocks
def windows(d, x0, y0, x1, y1, cw, ch, gap, rng, lit_p, lit_col, dark_col):
    y = y0 + gap
    while y + ch < y1:
        x = x0 + gap
        while x + cw < x1 - gap / 2:
            on = rng.random() < lit_p
            d.rectangle([x, y, x + cw, y + ch], fill=lit_col if on else dark_col)
            x += cw + gap
        y += ch + gap


def skyline(size, base_y, color, rng, hmin, hmax, wmin, wmax, lit_p=0.0,
            lit_col=(255, 214, 102, 235), dark_win=None, x_from=0, x_to=None, gap=(0, 5)):
    layer, d = new_layer(size)
    W = size[0]
    x = x_from - int(rng.integers(0, wmin))
    x_to = x_to or W
    while x < x_to:
        w = int(rng.integers(wmin, wmax))
        h = int(rng.integers(hmin, hmax))
        top = base_y - h
        d.rectangle([x, top, x + w, base_y + 4], fill=rgba(color))
        r = rng.random()
        if r < 0.20:                                  # antenna
            ax = x + w // 2
            d.rectangle([ax - 2, top - h * 0.25, ax + 2, top], fill=rgba(color))
        elif r < 0.38:                                # stepped crown
            d.rectangle([x + w * 0.2, top - h * 0.10, x + w * 0.8, top], fill=rgba(color))
            d.rectangle([x + w * 0.35, top - h * 0.18, x + w * 0.65, top - h * 0.10], fill=rgba(color))
        elif r < 0.48:                                # dome
            d.pieslice([x + w * 0.15, top - w * 0.35, x + w * 0.85, top + w * 0.35], 180, 360, fill=rgba(color))
        if lit_p > 0:
            cw, chh, g = max(4, w // 7), max(5, int(h / 22)), max(3, w // 12)
            windows(d, x, top + 6, x + w, base_y, cw, chh, g, rng, lit_p, lit_col,
                    dark_win or rgba(mix(color, (255, 255, 255), 0.07)))
        x += w + int(rng.integers(*gap))
    return layer


def trees(size, y_base, color, rng, x0, x1, n, scale):
    layer, d = new_layer(size)
    for _ in range(n):
        x = float(rng.uniform(x0, x1))
        r = float(rng.uniform(0.7, 1.3)) * scale
        d.rectangle([x - r * 0.08, y_base - r * 0.9, x + r * 0.08, y_base], fill=rgba(color))
        for dx, dy, rr in [(0, -1.5, 0.75), (-0.5, -1.15, 0.55), (0.5, -1.15, 0.55), (0, -1.0, 0.7)]:
            d.ellipse([x + dx * r - rr * r, y_base + dy * r - rr * r,
                       x + dx * r + rr * r, y_base + dy * r + rr * r], fill=rgba(color))
    return layer


def palm(d, x, y, h, col, lean=0.25):
    tx, ty = x + h * lean, y - h
    pts = [(x, y), (x + h * lean * 0.3, y - h * 0.5), (tx, ty)]
    d.line(pts, fill=col, width=max(3, int(h * 0.05)))
    # drooping, arching fronds
    for c in (-1.0, -.75, -.45, -.15, .15, .45, .75, 1.0):
        p1 = (tx + c * h * .26, ty - h * .15)
        p2 = (tx + c * h * .50, ty + h * (.02 + .16 * abs(c)))
        p3 = (tx + c * h * .58, ty + h * (.14 + .20 * abs(c)))
        d.line([(tx, ty), p1, p2, p3], fill=col, width=max(3, int(h * 0.04)), joint="curve")


# ------------------------------------------------------------------ landmarks
def lm_gate(d, cx, gy, u, col, rich=False):
    w, h = 7 * u, 5.2 * u
    d.rectangle([cx - w / 2, gy - h, cx + w / 2, gy], fill=col)
    d.rectangle([cx - w / 2 - .3 * u, gy - h - .35 * u, cx + w / 2 + .3 * u, gy - h], fill=col)
    for dx in (-w / 2 + .7 * u, w / 2 - .7 * u):
        d.pieslice([dx - .6 * u, gy - h - 1.1 * u, dx + .6 * u, gy - h + .25 * u], 180, 360, fill=col)
    d.pieslice([cx - .8 * u, gy - h - 1.7 * u, cx + .8 * u, gy - h + .0 * u], 180, 360, fill=col)
    ar = 1.45 * u
    d.rectangle([cx - ar, gy - 3.0 * u, cx + ar, gy], fill=HOLE)
    d.pieslice([cx - ar, gy - 3.0 * u - ar, cx + ar, gy - 3.0 * u + ar], 180, 360, fill=HOLE)
    for dx in (-2.5 * u, 2.5 * u):
        d.rectangle([cx + dx - .4 * u, gy - 2.2 * u, cx + dx + .4 * u, gy - 1.0 * u], fill=HOLE)
        d.pieslice([cx + dx - .4 * u, gy - 2.6 * u, cx + dx + .4 * u, gy - 1.8 * u], 180, 360, fill=HOLE)
    if rich:
        for dx in (-w / 2 - .4 * u, w / 2 + .4 * u):
            d.rectangle([cx + dx - .35 * u, gy - h - 1.4 * u, cx + dx + .35 * u, gy], fill=col)
            d.pieslice([cx + dx - .5 * u, gy - h - 2.1 * u, cx + dx + .5 * u, gy - h - .7 * u], 180, 360, fill=col)


def lm_minars(d, cx, gy, u, col):
    d.rectangle([cx - 3 * u, gy - 3.2 * u, cx + 3 * u, gy], fill=col)
    d.rectangle([cx - 2.5 * u, gy - 4.3 * u, cx + 2.5 * u, gy - 3.2 * u], fill=col)
    for k in range(-2, 3):                                            # parapet
        d.rectangle([cx + k * u - .22 * u, gy - 4.7 * u, cx + k * u + .22 * u, gy - 4.3 * u], fill=col)
    for dx in (-3 * u, 3 * u):
        d.rectangle([cx + dx - .38 * u, gy - 7.2 * u, cx + dx + .38 * u, gy], fill=col)
        d.rectangle([cx + dx - .55 * u, gy - 5.2 * u, cx + dx + .55 * u, gy - 4.9 * u], fill=col)
        d.pieslice([cx + dx - .6 * u, gy - 8.0 * u, cx + dx + .6 * u, gy - 6.6 * u], 180, 360, fill=col)
        d.rectangle([cx + dx - .04 * u, gy - 8.6 * u, cx + dx + .04 * u, gy - 7.8 * u], fill=col)
    for k in (-1.7, 0, 1.7):                                          # arches
        d.rectangle([cx + k * u - .6 * u, gy - 2.0 * u, cx + k * u + .6 * u, gy], fill=HOLE)
        d.pieslice([cx + k * u - .6 * u, gy - 2.6 * u, cx + k * u + .6 * u, gy - 1.4 * u], 180, 360, fill=HOLE)


def lm_gopuram(d, cx, gy, u, col):
    d.rectangle([cx - 3.4 * u, gy - 1.6 * u, cx + 3.4 * u, gy], fill=col)
    d.rectangle([cx - .7 * u, gy - 1.3 * u, cx + .7 * u, gy], fill=HOLE)
    y = gy - 1.6 * u
    for i in range(7):
        w = 3.0 * u * (1 - i * 0.11)
        d.polygon([(cx - w, y), (cx + w, y), (cx + w * .88, y - .85 * u), (cx - w * .88, y - .85 * u)], fill=col)
        n = max(2, int(5 - i * 0.5))
        for k in range(n):                                            # little niches
            nx = cx - w * .7 + k * (w * 1.4) / max(1, n - 1)
            d.rectangle([nx - .12 * u, y - .6 * u, nx + .12 * u, y - .25 * u], fill=HOLE)
        y -= .85 * u
    d.pieslice([cx - 1.0 * u, y - .8 * u, cx + 1.0 * u, y + .8 * u], 180, 360, fill=col)
    for dx in (-.6 * u, 0, .6 * u):
        d.ellipse([cx + dx - .12 * u, y - 1.3 * u, cx + dx + .12 * u, y - .85 * u], fill=col)


def lm_dome(d, cx, gy, u, col, columns=False):
    d.rectangle([cx - 4.2 * u, gy - 2.4 * u, cx + 4.2 * u, gy], fill=col)
    d.rectangle([cx - 1.4 * u, gy - 4.0 * u, cx + 1.4 * u, gy - 2.4 * u], fill=col)
    d.pieslice([cx - 1.9 * u, gy - 6.6 * u, cx + 1.9 * u, gy - 2.0 * u], 180, 360, fill=col)
    d.rectangle([cx - .12 * u, gy - 7.5 * u, cx + .12 * u, gy - 6.3 * u], fill=col)
    d.ellipse([cx - .22 * u, gy - 7.8 * u, cx + .22 * u, gy - 7.35 * u], fill=col)
    for dx in (-3.3 * u, 3.3 * u):
        d.rectangle([cx + dx - .6 * u, gy - 3.4 * u, cx + dx + .6 * u, gy - 2.4 * u], fill=col)
        d.pieslice([cx + dx - .8 * u, gy - 4.4 * u, cx + dx + .8 * u, gy - 2.7 * u], 180, 360, fill=col)
    if columns:
        d.polygon([(cx - 2.6 * u, gy - 2.4 * u), (cx + 2.6 * u, gy - 2.4 * u), (cx, gy - 3.4 * u)], fill=col)
        for k in range(-2, 3):
            d.rectangle([cx + k * .9 * u - .13 * u, gy - 2.4 * u, cx + k * .9 * u + .13 * u, gy], fill=HOLE)
    for k in range(-3, 4):
        d.rectangle([cx + k * 1.1 * u - .3 * u, gy - 1.6 * u, cx + k * 1.1 * u + .3 * u, gy - .2 * u], fill=HOLE)
        d.pieslice([cx + k * 1.1 * u - .3 * u, gy - 1.9 * u, cx + k * 1.1 * u + .3 * u, gy - 1.3 * u], 180, 360, fill=HOLE)


def lm_truss(d, cx, gy, u, col):
    half = 6.5 * u
    deck = gy - 1.0 * u
    d.rectangle([cx - half - u, deck, cx + half + u, deck + .35 * u], fill=col)
    top = lambda t: deck - .3 * u - 3.8 * u * (1 - t * t) ** 0.75
    n = 22
    pts = [(cx + (-1 + 2 * i / n) * half, top(-1 + 2 * i / n)) for i in range(n + 1)]
    d.line(pts, fill=col, width=int(.28 * u))
    for i in range(n + 1):
        x, y = pts[i]
        d.line([(x, y), (x, deck)], fill=col, width=int(.10 * u))
        if i < n:
            d.line([(x, deck), pts[i + 1]] if i % 2 == 0 else [(x, y), (pts[i + 1][0], deck)], fill=col, width=int(.08 * u))
    for dx in (-half, half):
        d.rectangle([cx + dx - .55 * u, deck, cx + dx + .55 * u, gy], fill=col)
    d.rectangle([cx - half - u, gy - .1 * u, cx + half + u, gy], fill=col)


def lm_cable(d, cx, gy, u, col):
    deck = gy - .9 * u
    d.rectangle([cx - 8 * u, deck, cx + 8 * u, deck + .3 * u], fill=col)
    for dx in (-2.2 * u, 2.2 * u):
        px = cx + dx
        d.polygon([(px - .55 * u, deck), (px - .1 * u, deck - 6.0 * u), (px + .1 * u, deck - 6.0 * u), (px + .55 * u, deck)], fill=col)
        d.rectangle([px - .45 * u, deck - 3.6 * u, px + .45 * u, deck - 3.35 * u], fill=col)
        for k in range(1, 9):
            tx = k * .85 * u
            d.line([(px, deck - 5.9 * u + k * .18 * u), (px - tx, deck)], fill=col, width=max(2, int(.05 * u)))
            d.line([(px, deck - 5.9 * u + k * .18 * u), (px + tx, deck)], fill=col, width=max(2, int(.05 * u)))
    d.rectangle([cx - 8 * u, gy - .1 * u, cx + 8 * u, gy], fill=col)


def lm_facade(d, cx, gy, u, col, pyramid=True):
    tiers = 5
    y = gy
    for i in range(tiers):
        w = (7.0 - i * 1.15 * (1 if pyramid else 0)) * u / 2
        h = 1.15 * u
        d.rectangle([cx - w, y - h, cx + w, y], fill=col)
        n = int(w * 2 / (0.62 * u))
        for k in range(n):
            wx = cx - w + (k + .5) * (2 * w / n)
            d.rectangle([wx - .13 * u, y - .85 * u, wx + .13 * u, y - .3 * u], fill=HOLE)
            d.pieslice([wx - .13 * u, y - 1.0 * u, wx + .13 * u, y - .7 * u], 180, 360, fill=HOLE)
        for dx in (-w + .35 * u, w - .35 * u):
            d.pieslice([cx + dx - .35 * u, y - h - .45 * u, cx + dx + .35 * u, y - h + .25 * u], 180, 360, fill=col)
        y -= h
    d.pieslice([cx - .5 * u, y - .8 * u, cx + .5 * u, y + .4 * u], 180, 360, fill=col)
    d.rectangle([cx - 1.0 * u, gy - 1.6 * u, cx + 1.0 * u, gy], fill=col)


def lm_qutub(d, cx, gy, u, col):
    H = 9 * u
    d.polygon([(cx - 1.1 * u, gy), (cx + 1.1 * u, gy), (cx + .5 * u, gy - H), (cx - .5 * u, gy - H)], fill=col)
    for k in range(1, 5):
        y = gy - H * k / 5
        wk = 1.1 * u - (.6 * u) * (k / 5)
        d.rectangle([cx - wk - .35 * u, y - .12 * u, cx + wk + .35 * u, y + .12 * u], fill=col)
        d.rectangle([cx - wk - .3 * u, y - .3 * u, cx - wk - .22 * u, y], fill=col)
        d.rectangle([cx + wk + .22 * u, y - .3 * u, cx + wk + .3 * u, y], fill=col)
    d.rectangle([cx - .6 * u, gy - H - .35 * u, cx + .6 * u, gy - H], fill=col)
    d.pieslice([cx - .5 * u, gy - H - 1.1 * u, cx + .5 * u, gy - H + .1 * u], 180, 360, fill=col)


def lm_mosque(d, cx, gy, u, col):
    d.rectangle([cx - 3.0 * u, gy - 2.4 * u, cx + 3.0 * u, gy], fill=col)
    d.pieslice([cx - 1.6 * u, gy - 4.8 * u, cx + 1.6 * u, gy - .0 * u], 180, 360, fill=col)
    d.pieslice([cx - 2.9 * u, gy - 3.7 * u, cx - 1.5 * u, gy - 1.3 * u], 180, 360, fill=col)
    d.pieslice([cx + 1.5 * u, gy - 3.7 * u, cx + 2.9 * u, gy - 1.3 * u], 180, 360, fill=col)
    for dx in (-3.7 * u, 3.7 * u):
        d.rectangle([cx + dx - .32 * u, gy - 6.2 * u, cx + dx + .32 * u, gy], fill=col)
        for yy in (-3.0, -4.4):
            d.rectangle([cx + dx - .6 * u, gy + yy * u, cx + dx + .6 * u, gy + (yy + .18) * u], fill=col)
        d.pieslice([cx + dx - .45 * u, gy - 7.0 * u, cx + dx + .45 * u, gy - 5.6 * u], 180, 360, fill=col)
    for k in (-1.8, 0, 1.8):
        d.rectangle([cx + k * u - .5 * u, gy - 1.6 * u, cx + k * u + .5 * u, gy], fill=HOLE)
        d.pieslice([cx + k * u - .5 * u, gy - 2.1 * u, cx + k * u + .5 * u, gy - 1.1 * u], 180, 360, fill=HOLE)


def lm_towers(d, cx, gy, u, col, rng, n=5, spread=7.5):
    xs = np.linspace(cx - spread * u / 2, cx + spread * u / 2, n)
    for i, x in enumerate(xs):
        w = float(rng.uniform(1.2, 1.9)) * u
        h = float(rng.uniform(5.5, 9.5)) * u * (1.0 if i % 2 else .8)
        d.rectangle([x - w / 2, gy - h, x + w / 2, gy], fill=col)
        k = i % 3
        if k == 0:
            d.polygon([(x - w / 2, gy - h), (x + w / 2, gy - h), (x, gy - h - 1.1 * u)], fill=col)
        elif k == 1:
            d.rectangle([x - w * .3, gy - h - .6 * u, x + w * .3, gy - h], fill=col)
            d.rectangle([x - .05 * u, gy - h - 1.6 * u, x + .05 * u, gy - h - .6 * u], fill=col)
        else:
            d.rectangle([x - w / 2 - .1 * u, gy - h - .25 * u, x + w / 2 + .1 * u, gy - h], fill=col)
        for j in range(1, 9):                                          # glass reflection bands
            yy = gy - h + j * h / 9
            d.line([(x - w / 2, yy + .5 * u), (x + w / 2, yy - .1 * u)], fill=HOLE, width=max(1, int(.05 * u)))


def lm_fort(d, cx, gy, u, col):
    d.rectangle([cx - 5 * u, gy - 2.2 * u, cx + 5 * u, gy], fill=col)
    for k in range(-9, 10):
        d.rectangle([cx + k * .5 * u - .17 * u, gy - 2.6 * u, cx + k * .5 * u + .17 * u, gy - 2.2 * u], fill=col)
    for dx in (-5 * u, 5 * u):
        d.rectangle([cx + dx - .8 * u, gy - 3.6 * u, cx + dx + .8 * u, gy], fill=col)
        d.pieslice([cx + dx - .95 * u, gy - 4.5 * u, cx + dx + .95 * u, gy - 3.0 * u], 180, 360, fill=col)
    d.rectangle([cx - 1.2 * u, gy - 3.6 * u, cx + 1.2 * u, gy], fill=col)
    d.rectangle([cx - .75 * u, gy - 1.9 * u, cx + .75 * u, gy], fill=HOLE)
    d.pieslice([cx - .75 * u, gy - 2.65 * u, cx + .75 * u, gy - 1.15 * u], 180, 360, fill=HOLE)
    d.pieslice([cx - 1.2 * u, gy - 4.5 * u, cx + 1.2 * u, gy - 2.8 * u], 180, 360, fill=col)


def lm_nets(d, cx, gy, u, col):
    for i, dx in enumerate((-4.2 * u, 0, 4.2 * u)):
        bx = cx + dx
        tip = (bx - .6 * u, gy - 4.6 * u)
        d.line([(bx + .3 * u, gy), tip], fill=col, width=max(4, int(.22 * u)))
        d.line([(bx - 2.5 * u, gy), tip], fill=col, width=max(3, int(.15 * u)))
        d.line([tip, (bx - 4.2 * u, gy - 2.9 * u)], fill=col, width=max(2, int(.08 * u)))
        for k in range(8):                                             # hanging net
            nx = bx - 4.2 * u + k * .38 * u
            d.line([(nx, gy - 2.9 * u + k * .04 * u), (nx - .1 * u, gy - 1.2 * u)], fill=col, width=max(1, int(.04 * u)))
        d.line([(bx - 4.2 * u, gy - 1.2 * u), (bx - 1.4 * u, gy - 1.1 * u)], fill=col, width=max(2, int(.06 * u)))
    for dx in (-6.5 * u, 6.5 * u):
        palm(d, cx + dx, gy, 3.6 * u, col)


def lm_stadium(d, cx, gy, u, col):
    d.pieslice([cx - 5.5 * u, gy - 3.2 * u, cx + 5.5 * u, gy + 3.2 * u], 180, 360, fill=col)
    d.pieslice([cx - 4.7 * u, gy - 2.4 * u, cx + 4.7 * u, gy + 2.6 * u], 180, 360, fill=HOLE)
    d.rectangle([cx - 5.5 * u, gy - .15 * u, cx + 5.5 * u, gy], fill=col)
    for k, dx in enumerate((-6.4 * u, -2.2 * u, 2.2 * u, 6.4 * u)):
        h = (6.8 if k in (0, 3) else 5.6) * u
        d.rectangle([cx + dx - .12 * u, gy - h, cx + dx + .12 * u, gy], fill=col)
        d.rectangle([cx + dx - .9 * u, gy - h - .6 * u, cx + dx + .9 * u, gy - h], fill=col)
        for j in range(4):
            lx = cx + dx - .75 * u + j * .5 * u
            d.ellipse([lx - .08 * u, gy - h - .5 * u, lx + .08 * u, gy - h - .1 * u], fill=HOLE)


def lm_hand(d, cx, gy, u, col):
    d.rectangle([cx - .13 * u, gy - 7 * u, cx + .13 * u, gy], fill=col)
    d.rectangle([cx - 1.2 * u, gy - .6 * u, cx + 1.2 * u, gy], fill=col)
    top = gy - 7 * u
    d.polygon([(cx + .1 * u, top + .3 * u), (cx + 2.4 * u, top - .7 * u), (cx + 2.9 * u, top - 2.6 * u),
               (cx + 1.5 * u, top - 3.3 * u), (cx + .1 * u, top - 1.5 * u)], fill=col)
    d.polygon([(cx - .1 * u, top + .3 * u), (cx - 2.0 * u, top - .5 * u), (cx - 2.4 * u, top - 2.0 * u),
               (cx - .1 * u, top - 1.6 * u)], fill=col)
    for lo in range(4):                                                # low-rise Corbusier blocks
        bx = cx - 7 * u + lo * 2.0 * u if lo < 2 else cx + 3.2 * u + (lo - 2) * 2.4 * u
        d.rectangle([bx, gy - 1.4 * u, bx + 1.7 * u, gy], fill=col)


def mountains(size, base_y, color, rng, hmin, hmax, step):
    layer, d = new_layer(size)
    pts = [(0, base_y)]
    x = 0
    while x < size[0] + step:
        pts.append((x, base_y - float(rng.uniform(hmin, hmax))))
        x += step
    pts += [(size[0], base_y), (0, base_y)]
    d.polygon(pts, fill=rgba(color))
    return layer.filter(ImageFilter.GaussianBlur(1.2 * S))


# ------------------------------------------------------------------ scene composer
def compose_scene(W, H, pal, rng, landmarks, water=False, night=False, sun_x=.72,
                  ground=.80, tree_fg=True, towers_bg=True, hills=False, dark_bottom=.38):
    top, mid, hor = (hx(c) for c in pal)
    size = (W, H)
    gy = int(H * ground)
    base = vgradient(W, H, [(0, top), (.55, mid), (min(.98, ground + .05), hor), (1, hor)])

    base = over(base, glow_layer(size, W * sun_x, gy - H * .05, W * .42, mix(hor, (255, 255, 255), .25), .75))
    sc = new_layer(size)
    sd = sc[1]
    sd.ellipse([W * sun_x - H * .055, gy - H * .105, W * sun_x + H * .055, gy + H * .005],
               fill=rgba(mix(hor, (255, 255, 255), .55), 235))
    base = over(base, sc[0])

    if night or True:                                                  # faint stars on the dark top
        st, sdraw = new_layer(size)
        for _ in range(110):
            x, y = rng.uniform(0, W), rng.uniform(0, H * .38)
            r = float(rng.uniform(.8, 2.1)) * S / 2
            sdraw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, int(rng.uniform(60, 190))))
        base = over(base, st)

    far = mix(mid, top, .45)
    midc = mix(top, (0, 0, 0), .35)
    near = mix(top, (0, 0, 0), .68)

    if hills:
        base = over(base, mountains(size, gy, mix(mid, top, .55), rng, H * .10, H * .30, W / 9))
        base = over(base, mountains(size, gy, mix(top, (0, 0, 0), .25), rng, H * .06, H * .18, W / 12))
    if towers_bg:
        base = over(base, skyline(size, gy, far, rng, int(H * .10), int(H * .26), int(W * .03), int(W * .06)).filter(ImageFilter.GaussianBlur(1.5 * S)), .85)
        base = over(base, skyline(size, gy, midc, rng, int(H * .08), int(H * .30), int(W * .035), int(W * .07),
                                  lit_p=.16, lit_col=rgba(hx("#FFD166"), 210)))
    lay, d = new_layer(size)
    for fn, xf, uf, kw in landmarks:
        if fn is lm_towers:
            fn(d, W * xf, gy, H * uf, rgba(near), rng, **kw)
        else:
            fn(d, W * xf, gy, H * uf, rgba(near), **kw)
    base = over(base, lay)

    gl, gd = new_layer(size)
    if water:
        wat = vgradient(W, H - gy, [(0, mix(hor, mid, .35)), (.4, mix(mid, top, .55)), (1, mix(top, (0, 0, 0), .55))])
        base.paste(wat, (0, gy))
        rl, rd = new_layer(size)
        for _ in range(60):
            x = rng.uniform(0, W)
            y = rng.uniform(gy + 6, H)
            ww = rng.uniform(.03, .16) * W * (1 - (y - gy) / (H - gy) * .4)
            near_sun = max(0, 1 - abs(x - W * sun_x) / (W * .4))
            rd.rectangle([x, y, x + ww, y + 2 * S], fill=rgba(mix(hor, (255, 255, 255), .5), 55 + 140 * near_sun))
        base = over(base, rl if False else rl)
        base = over(base, rl)
        gd.rectangle([0, gy - 3, W, gy + 6 * S], fill=rgba(near))
    else:
        gd.rectangle([0, gy, W, H], fill=rgba(near))
    base = over(base, gl)

    if tree_fg:
        base = over(base, trees(size, gy + 8 * S, near, rng, 0, W * .22, 5, H * .045))
        base = over(base, trees(size, gy + 8 * S, near, rng, W * .80, W, 5, H * .05))

    shade = Image.new("RGBA", size, (0, 0, 0, 0))
    ys = np.linspace(0, 1, H)
    a = np.clip((ys - (1 - dark_bottom)) / dark_bottom, 0, 1) ** 1.6 * 175
    arr = np.zeros((H, W, 4), dtype=np.uint8)
    arr[:, :, 3] = a[:, None].astype(np.uint8)
    shade = Image.fromarray(arr, "RGBA")
    return over(base, shade)


# ------------------------------------------------------------------ city catalogue
def _L(fn, x, u, **kw):
    return (fn, x, u, kw)


CITY_SCENES = {
    "Mumbai":     dict(pal=("#14213D", "#8E2C6A", "#F59E0B"), water=True, lm=[_L(lm_cable, .24, .050), _L(lm_gate, .68, .062, rich=False)]),
    "Delhi":      dict(pal=("#2B1B4B", "#B5456B", "#FFB347"), lm=[_L(lm_gate, .40, .060), _L(lm_qutub, .78, .048)]),
    "Gurugram":   dict(pal=("#0B3C5D", "#328CC1", "#D9E7F2"), lm=[_L(lm_towers, .55, .062, n=6, spread=11)]),
    "Noida":      dict(pal=("#0F4C75", "#3282B8", "#BBE1FA"), lm=[_L(lm_towers, .45, .056, n=5, spread=9), _L(lm_towers, .85, .040, n=2, spread=2.5)]),
    "Bengaluru":  dict(pal=("#064E3B", "#10B981", "#FEF3C7"), lm=[_L(lm_dome, .50, .055, columns=True)]),
    "Hyderabad":  dict(pal=("#2E1065", "#7C3AED", "#FBBF24"), lm=[_L(lm_minars, .52, .058)]),
    "Pune":       dict(pal=("#1E3A8A", "#F472B6", "#FDE68A"), hills=True, lm=[_L(lm_fort, .55, .060)]),
    "Chennai":    dict(pal=("#0C4A6E", "#0EA5E9", "#FCD34D"), water=True, lm=[_L(lm_gopuram, .52, .056)]),
    "Kolkata":    dict(pal=("#1F2937", "#B45309", "#FCD34D"), water=True, lm=[_L(lm_truss, .58, .062), _L(lm_dome, .16, .030)]),
    "Ahmedabad":  dict(pal=("#7C2D12", "#EA580C", "#FDE68A"), lm=[_L(lm_mosque, .52, .060)]),
    "Chandigarh": dict(pal=("#134E4A", "#14B8A6", "#E6FFFA"), hills=True, lm=[_L(lm_hand, .55, .056)]),
    "Mohali":     dict(pal=("#1E1B4B", "#4F46E5", "#C7D2FE"), lm=[_L(lm_stadium, .50, .062), _L(lm_towers, .90, .040, n=2, spread=2)]),
    "Jaipur":     dict(pal=("#831843", "#EC4899", "#FDE68A"), lm=[_L(lm_facade, .52, .060, pyramid=True)]),
    "Lucknow":    dict(pal=("#78350F", "#D97706", "#FEF3C7"), lm=[_L(lm_gate, .52, .066, rich=True)]),
    "Kochi":      dict(pal=("#083344", "#06B6D4", "#FB923C"), water=True, lm=[_L(lm_nets, .52, .060)]),
    "Indore":     dict(pal=("#7F1D1D", "#DC2626", "#FDBA74"), lm=[_L(lm_facade, .52, .060, pyramid=False)]),
}


def save_jpg(img, path, size, q=86):
    img = img.convert("RGB").resize(size, Image.LANCZOS)
    img.save(path, quality=q, optimize=True)


def make_city_cards():
    for name, cfg in CITY_SCENES.items():
        rng = np.random.default_rng(zlib.crc32(name.encode()))
        sc = compose_scene(640 * S, 400 * S, cfg["pal"], rng, cfg["lm"], water=cfg.get("water", False),
                           hills=cfg.get("hills", False), sun_x=float(rng.uniform(.25, .78)))
        save_jpg(sc, C.ASSET_DIR / f"city_{name.lower()}.jpg", (640, 400))
        print("  city", name)


def make_hero():
    rng = np.random.default_rng(7)
    W, H = 1700 * S, 520 * S
    lm = [_L(lm_gate, .22, .060), _L(lm_qutub, .36, .050), _L(lm_towers, .62, .062, n=7, spread=13),
          _L(lm_dome, .86, .050, columns=False), _L(lm_towers, .97, .045, n=2, spread=2)]
    sc = compose_scene(W, H, ("#0B1F3A", "#4C2A85", "#F59E0B"), rng, lm, sun_x=.5, towers_bg=True,
                       tree_fg=False, dark_bottom=.30)
    save_jpg(sc, C.ASSET_DIR / "hero.jpg", (1700, 520), q=84)
    print("  hero")


# ------------------------------------------------------------------ property type art
def _day_sky(W, H, top="#7CC4F0", bot="#F4F9FF", ground=.84, rng=None):
    base = vgradient(W, H, [(0, hx(top)), (ground, hx(bot)), (1, hx(bot))])
    cl, cd = new_layer((W, H))
    for _ in range(5):
        x, y = rng.uniform(0, W), rng.uniform(H * .05, H * .35)
        for dx, dy, r in [(0, 0, 38), (45, 6, 30), (-45, 8, 28), (15, -16, 30)]:
            cd.ellipse([x + dx * S - r * S, y + dy * S - r * S * .6, x + dx * S + r * S, y + dy * S + r * S * .6], fill=(255, 255, 255, 190))
    return over(base, cl.filter(ImageFilter.GaussianBlur(3 * S)))


def _ground(base, W, H, y, col="#3E7C4A"):
    l, d = new_layer((W, H))
    d.rectangle([0, y, W, H], fill=rgba(hx(col)))
    d.rectangle([0, y + (H - y) * .55, W, H], fill=rgba(hx("#3B3F4A")))
    for k in range(0, W, 60 * S):
        d.rectangle([k, y + (H - y) * .78, k + 30 * S, y + (H - y) * .82], fill=(240, 240, 240, 220))
    return over(base, l)


def _tower(d, x, gy, w, h, body, glass_a, glass_b, rng, balcony=True):
    d.rectangle([x, gy - h, x + w, gy], fill=rgba(hx(body)))
    rows = int(h / (16 * S))
    cols = max(2, int(w / (22 * S)))
    cw = w / cols
    for r in range(rows):
        for c in range(cols):
            col = hx(glass_a) if rng.random() > .18 else hx(glass_b)
            x0 = x + c * cw + 4 * S
            y0 = gy - h + r * 16 * S + 4 * S
            d.rectangle([x0, y0, x0 + cw - 8 * S, y0 + 9 * S], fill=rgba(col))
        if balcony:
            d.rectangle([x, gy - h + r * 16 * S + 13 * S, x + w, gy - h + r * 16 * S + 15 * S], fill=(255, 255, 255, 120))
    d.rectangle([x + w * .4, gy - h - 12 * S, x + w * .6, gy - h], fill=rgba(hx(body)))


def art_apartment(rng):
    W, H = 520 * S, 340 * S
    gy = int(H * .84)
    base = _ground(_day_sky(W, H, rng=rng), W, H, gy)
    l, d = new_layer((W, H))
    _tower(d, W * .08, gy, W * .20, H * .46, "#C9D6E8", "#7FB3D5", "#F6D365", rng)
    _tower(d, W * .72, gy, W * .20, H * .52, "#D5DCE8", "#7FB3D5", "#F6D365", rng)
    _tower(d, W * .30, gy, W * .38, H * .74, "#EDF2F8", "#4A90C2", "#FFD166", rng)
    base = over(base, l)
    base = over(base, trees((W, H), gy + 4 * S, hx("#2E6B3A"), rng, 0, W, 11, 20 * S))
    return base


def art_floor(rng):
    W, H = 520 * S, 340 * S
    gy = int(H * .84)
    base = _ground(_day_sky(W, H, "#9AD0F0", "#FFF7E8", rng=rng), W, H, gy)
    l, d = new_layer((W, H))
    for i, (x, w, col) in enumerate([(.02, .27, "#E9D8C3"), (.36, .30, "#F4E4D1"), (.72, .26, "#D9C7B2")]):
        top = gy - H * (.58 if i == 1 else .50)
        d.rectangle([W * x, top, W * (x + w), gy], fill=rgba(hx(col)))
        d.rectangle([W * x - 4 * S, top - 8 * S, W * (x + w) + 4 * S, top], fill=rgba(hx("#8B6B4A")))
        for fl in range(4 if i == 1 else 3):
            y0 = top + 12 * S + fl * (gy - top - 12 * S) / (4 if i == 1 else 3)
            hgt = (gy - top - 12 * S) / (4 if i == 1 else 3)
            for c in range(2):
                wx = W * x + W * w * (.12 + c * .46)
                d.rectangle([wx, y0 + 6 * S, wx + W * w * .32, y0 + hgt - 14 * S], fill=rgba(hx("#6FA8D6")))
                d.rectangle([wx - 3 * S, y0 + hgt - 12 * S, wx + W * w * .32 + 3 * S, y0 + hgt - 8 * S], fill=rgba(hx("#5B4636")))
                for k in range(7):
                    px = wx - 2 * S + k * (W * w * .36) / 6
                    d.rectangle([px, y0 + hgt - 22 * S, px + 2 * S, y0 + hgt - 12 * S], fill=rgba(hx("#5B4636")))
    base = over(base, l)
    cl, cd = new_layer((W, H))
    cx = W * .50
    cd.rounded_rectangle([cx - 52 * S, gy - 8 * S, cx + 52 * S, gy + 34 * S], radius=14 * S, fill=rgba(hx("#C2410C")))
    cd.rounded_rectangle([cx - 30 * S, gy - 26 * S, cx + 30 * S, gy - 4 * S], radius=10 * S, fill=rgba(hx("#C2410C")))
    cd.rectangle([cx - 22 * S, gy - 22 * S, cx + 22 * S, gy - 8 * S], fill=rgba(hx("#BFE3F7")))
    for dx in (-30 * S, 30 * S):
        cd.ellipse([cx + dx - 12 * S, gy + 20 * S, cx + dx + 12 * S, gy + 44 * S], fill=rgba(hx("#1F2937")))
    base = over(base, cl)
    return over(base, trees((W, H), gy + 4 * S, hx("#2E6B3A"), rng, 0, W, 6, 22 * S))


def art_house(rng):
    W, H = 520 * S, 340 * S
    gy = int(H * .82)
    base = _ground(_day_sky(W, H, "#8FCBF0", "#FFF3D6", rng=rng), W, H, gy, "#4C8B4F")
    l, d = new_layer((W, H))
    x0, x1 = W * .22, W * .78
    top = gy - H * .30
    d.rectangle([x0, top, x1, gy], fill=rgba(hx("#F3E5CB")))
    d.polygon([(x0 - 22 * S, top + 2 * S), (W * .5, top - H * .26), (x1 + 22 * S, top + 2 * S)], fill=rgba(hx("#B4532A")))
    d.rectangle([W * .64, top - H * .22, W * .68, top - H * .06], fill=rgba(hx("#8C3F1F")))
    d.rectangle([W * .46, gy - H * .20, W * .54, gy], fill=rgba(hx("#6B4226")))
    for wx in (.28, .60):
        d.rectangle([W * wx, top + H * .06, W * (wx + .12), top + H * .20], fill=rgba(hx("#7DB9E8")))
        d.rectangle([W * wx + W * .06 - 1 * S, top + H * .06, W * wx + W * .06 + 1 * S, top + H * .20], fill=rgba(hx("#F3E5CB")))
        d.rectangle([W * wx - 3 * S, top + H * .20, W * (wx + .12) + 3 * S, top + H * .215], fill=rgba(hx("#8C3F1F")))
    d.rectangle([W * .16, gy - H * .09, W * .84, gy + 2 * S], fill=rgba(hx("#E7D3B0")))
    for k in range(int(W * .16), int(W * .84), 14 * S):
        d.rectangle([k, gy - H * .12, k + 3 * S, gy - H * .09], fill=rgba(hx("#E7D3B0")))
    d.polygon([(W * .46, gy), (W * .54, gy), (W * .60, H), (W * .40, H)], fill=rgba(hx("#D9C5A0")))
    base = over(base, l)
    base = over(base, trees((W, H), gy + 2 * S, hx("#2E6B3A"), rng, 0, W * .2, 3, 30 * S))
    return over(base, trees((W, H), gy + 2 * S, hx("#2E6B3A"), rng, W * .8, W, 3, 30 * S))


def art_villa(rng):
    W, H = 520 * S, 340 * S
    gy = int(H * .70)
    base = vgradient(W, H, [(0, hx("#6FB7E9")), (.7, hx("#FDF3E0")), (1, hx("#6FB7E9"))])
    cl, cd = new_layer((W, H))
    cd.rectangle([0, gy, W, H], fill=rgba(hx("#5C9A57")))
    base = over(base, cl)
    l, d = new_layer((W, H))
    d.rectangle([W * .10, gy - H * .26, W * .52, gy], fill=rgba(hx("#F6F4EF")))
    d.rectangle([W * .40, gy - H * .50, W * .88, gy], fill=rgba(hx("#ECE8E0")))
    d.rectangle([W * .06, gy - H * .27, W * .56, gy - H * .245], fill=rgba(hx("#2F3A45")))
    d.rectangle([W * .36, gy - H * .51, W * .92, gy - H * .48], fill=rgba(hx("#2F3A45")))
    d.rectangle([W * .13, gy - H * .20, W * .36, gy - H * .02], fill=rgba(hx("#6EB6E6")))
    d.rectangle([W * .44, gy - H * .42, W * .84, gy - H * .26], fill=rgba(hx("#6EB6E6")))
    d.rectangle([W * .44, gy - H * .22, W * .60, gy - H * .02], fill=rgba(hx("#A9744A")))
    for k in range(7):
        d.rectangle([W * .44 + k * W * .025, gy - H * .22, W * .44 + k * W * .025 + 2 * S, gy - H * .02], fill=rgba(hx("#8A5A36")))
    d.line([(W * .64, gy - H * .42), (W * .84, gy - H * .26)], fill=(255, 255, 255, 140), width=3 * S)
    base = over(base, l)
    pl, pd_ = new_layer((W, H))
    pd_.polygon([(W * .14, gy + H * .06), (W * .70, gy + H * .06), (W * .78, H * .96), (W * .06, H * .96)], fill=rgba(hx("#E8E3D8")))
    pd_.polygon([(W * .17, gy + H * .09), (W * .67, gy + H * .09), (W * .74, H * .93), (W * .10, H * .93)], fill=rgba(hx("#27B6D3")))
    for k in range(9):
        y = gy + H * .12 + k * H * .075
        pd_.line([(W * .2 + (k % 3) * 25 * S, y), (W * .32 + (k % 3) * 25 * S, y)], fill=(255, 255, 255, 120), width=2 * S)
    base = over(base, pl)
    pa, pdr = new_layer((W, H))
    palm(pdr, W * .93, gy + H * .03, H * .36, rgba(hx("#2E6B3A")), lean=-.10)
    palm(pdr, W * .05, gy + H * .04, H * .30, rgba(hx("#2E6B3A")), lean=.15)
    return over(base, pa)


def art_penthouse(rng):
    W, H = 520 * S, 340 * S
    base = vgradient(W, H, [(0, hx("#0B1F3A")), (.5, hx("#6D3F8F")), (.82, hx("#F59E0B")), (1, hx("#F59E0B"))])
    base = over(base, glow_layer((W, H), W * .75, H * .78, W * .5, (255, 200, 120), .6))
    base = over(base, skyline((W, H), int(H * .80), mix(hx("#0B1F3A"), (60, 40, 90), .3), rng, int(H * .12), int(H * .40),
                              int(W * .04), int(W * .08), lit_p=.35))
    l, d = new_layer((W, H))
    top = H * .60
    d.rectangle([0, top, W, H], fill=rgba(hx("#161C2D")))
    d.rectangle([0, top - 8 * S, W, top + 4 * S], fill=rgba(hx("#2A3350")))
    d.rectangle([W * .18, top - H * .22, W * .70, top - 8 * S], fill=rgba(hx("#FFD98A"), 235))
    d.rectangle([W * .16, top - H * .235, W * .72, top - H * .21], fill=rgba(hx("#10141F")))
    for k in range(1, 6):
        d.rectangle([W * .18 + k * W * .52 / 6 - 1.5 * S, top - H * .22, W * .18 + k * W * .52 / 6 + 1.5 * S, top - 8 * S], fill=rgba(hx("#10141F")))
    d.rectangle([W * .02, top - H * .09, W * .16, top - 8 * S], fill=(255, 255, 255, 25))
    for k in range(40):
        x = W * .02 + k * (W * .96) / 40
        d.rectangle([x, top - H * .10, x + 2 * S, top - 8 * S], fill=rgba(hx("#8F9BB8")))
    d.rectangle([W * .02, top - H * .105, W * .98, top - H * .09], fill=rgba(hx("#8F9BB8")))
    for k in range(18):                                                # string lights
        x = W * .08 + k * W * .05
        d.ellipse([x - 3 * S, top - H * .28 + (k % 2) * 3 * S, x + 3 * S, top - H * .28 + 6 * S + (k % 2) * 3 * S], fill=rgba(hx("#FFE08A")))
    for px in (.76, .88):
        d.rectangle([W * px, top - H * .20, W * px + 16 * S, top - H * .12], fill=rgba(hx("#3B2A22")))
        d.ellipse([W * px - 10 * S, top - H * .30, W * px + 26 * S, top - H * .17], fill=rgba(hx("#2E6B3A")))
    return over(base, l)


def make_type_art():
    rng = np.random.default_rng(11)
    for name, fn in [("apartment", art_apartment), ("builder_floor", art_floor), ("independent_house", art_house),
                     ("villa", art_villa), ("penthouse", art_penthouse)]:
        save_jpg(fn(rng), C.ASSET_DIR / f"type_{name}.jpg", (520, 340), q=88)
        print("  type", name)


def make_logo():
    W = H = 256 * S
    base = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    grad = vgradient(W, H, [(0, hx("#0E7C86")), (1, hx("#0B1F3A"))])
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, W - 1, H - 1], radius=58 * S, fill=255)
    base.paste(grad, (0, 0), mask)
    l, d = new_layer((W, H))
    white = (255, 255, 255, 255)
    gold = rgba(hx("#F59E0B"))
    cx = W / 2
    d.polygon([(cx, 54 * S), (cx + 82 * S, 118 * S), (cx - 82 * S, 118 * S)], fill=white)
    d.rectangle([cx - 62 * S, 116 * S, cx + 62 * S, 206 * S], fill=white)
    d.rectangle([cx - 66 * S, 96 * S, cx - 46 * S, 128 * S], fill=white)
    # rupee sign cut-out in gold
    x0, x1 = cx - 24 * S, cx + 24 * S
    d.line([(x0, 138 * S), (x1, 138 * S)], fill=gold, width=7 * S)
    d.line([(x0, 156 * S), (x1, 156 * S)], fill=gold, width=7 * S)
    d.arc([x0 - 8 * S, 138 * S, x1 - 2 * S, 174 * S], 270, 90, fill=gold, width=7 * S)
    d.line([(x0 + 2 * S, 172 * S), (x1 - 2 * S, 198 * S)], fill=gold, width=7 * S)
    base = over(base, l)
    base.resize((256, 256), Image.LANCZOS).save(C.ASSET_DIR / "logo.png", optimize=True)
    print("  logo")


def main():
    C.ASSET_DIR.mkdir(parents=True, exist_ok=True)
    print("Drawing images ...")
    make_hero()
    make_city_cards()
    make_type_art()
    make_logo()
    print("Done ->", C.ASSET_DIR)


if __name__ == "__main__":
    main()
