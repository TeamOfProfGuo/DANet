import pickle
import numpy as np
from PIL import Image, ImageOps, ImageFilter

NETID = 'hl3797'
# MASK_GT_PATH = '/gpfsnyu/scratch/' + NETID + '/dataset/NYUD_v2/nyu_labels40/'
MASK_GT_PATH = './pred_masks/'

def rgb_to_idx(rgb):
    mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            for k in range(45):
                if rgb[i, j, :].all() == COLORS[k].all():
                    mask[i, j] = k
    return mask

def show_mask_classes(idx = 0):
    img = np.array(Image.open(MASK_GT_PATH + str(idx) + '.png'))
    mask1 = rgb_to_idx(img[481:, :481, :])
    mask2 = rgb_to_idx(img[481:, 481:, :])
    unique1, counts1 = np.unique(mask1, return_counts=True)
    freq1 = np.asarray((unique1, counts1)).T
    unique2, counts2 = np.unique(mask2, return_counts=True)
    freq2 = np.asarray((unique2, counts2)).T
    return freq1, freq2

def generate_colors():
    colors = []
    t = 255 * 0.2
    for i in range(1, 5):
        for j in range(1, 5):
            for k in range(1, 5):
                colors.append(np.array([t * i, t * j, t * k], dtype=np.uint8))
    while len(colors) <= 256:
        colors.append(np.array([0, 0, 0], dtype=np.uint8))
    return colors

def show_dat():
    data = pickle.load(open('./batch1.dat', 'rb'))
    # t = data['target'][0].numpy().astype(np.uint8)
    # # t = t.reshape(t.shape[0], t.shape[1], 1).astype(np.uint8)
    # print(t.shape, t.dtype)
    # img = Image.fromarray(t)
    # img.save('./test_mask.png', 'png')
    img = np.transpose((data['img'][0].numpy() * 255).astype(np.uint8), (1, 2, 0))
    dep = data['dep'][0]
    t1 = data['target'][0]
    t2 = data['pred'][0]
    img1 = tensor_to_rgb(t1)
    # img1.show()
    img2 = tensor_to_rgb(t2)
    # img2.show()
    dep_size = dep.size()
    dep_3c = np.transpose((dep.expand(3, dep_size[1], dep_size[2]).numpy() * 255).astype(np.uint8), (1, 2, 0))
    # print(img.shape, dep_3c.shape, img1.shape, img2.shape)
    part1 = np.concatenate((img, dep_3c), axis = 1)
    part2 = np.concatenate((img1, img2), axis = 1)
    res = np.concatenate((part1, part2), axis = 0)
    print(res.shape)
    ans = Image.fromarray(res)
    ans.show()


def tensor_to_rgb(t):
    assert len(t.shape) == 2
    colors = generate_colors()
    t = t.numpy().astype(np.uint8)
    rgb = np.zeros((t.shape[0], t.shape[1], 3), dtype=np.uint8)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            rgb[i, j, :] = colors[t[i, j]]
    return rgb # Image.fromarray(rgb)

def draw_legend():
    legend = np.ones((500, 1200, 3), dtype=np.uint8) * 255
    block_size = (40, 40)
    class_count = 0
    x_gap = 50
    y_gap = 80
    for i in range(5):
        for j in range(9):
            start_x = i * block_size[0] + (i + 1) * x_gap
            end_x = start_x + block_size[0]
            start_y = j * block_size[1] + (j + 1) * y_gap
            end_y = start_y + block_size[1]
            for x in range(start_x, end_x+1):
                for y in range(start_y, end_y+1):
                    legend[x, y, :] = COLORS[class_count]
            class_count += 1
            if class_count == 42:
                break
    return Image.fromarray(legend)

if __name__ == "__main__":
    COLORS = generate_colors()

    print(show_mask_classes(idx = 16))
    # show_dat()

    # legend = draw_legend()
    # legend.show()