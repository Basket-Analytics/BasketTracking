import cv2
import matplotlib.pyplot as plt


def plt_plot(img, title=None, cmap='viridis', additional_points=None):
    plt.figure(figsize=(16, 8))
    plt.title(f"{title + ': ' if title is not None else ''}{img.shape}")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if (cmap != 'gray') else img, cmap=cmap)
    if additional_points is not None:
        [plt.plot(p[0], p[1], 'ro') for p in additional_points]
    plt.tight_layout()
    plt.show()
