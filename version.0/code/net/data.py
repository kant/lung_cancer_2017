from net.common import *

# draw functions ---------

def imshow(name, image, resize=None):

    height = image.shape[0]
    width  = image.shape[1]
    if resize is not None:
        height *= resize
        width  *= resize

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.resizeWindow(name, width, height)


def data_to_color_img(data, scale=1, offset=(0,0,0), is_auto=False):
    if is_auto:
        min = np.min(data.reshape(-1))
        max = np.max(data.reshape(-1))
        scale  = 255/(max - min)
        offset = (-255 * min/(max - min),)*3

    if np.ndim(data)==2:
        data = data[:, :, None] * np.ones(3)[None, None, :]

    image = scale * data + np.array(offset)
    image = image.astype(np.uint8)
    return image


def data_to_gray_img(data, scale=255, offset=0, is_auto=False):
    if is_auto:
        min = np.min(data.reshape(-1))
        max = np.max(data.reshape(-1))
        scale  = 255/(max - min)
        offset = -255 * min/(max - min)

    image = scale * data + offset
    image = image.astype(np.uint8)
    return image


def draw_contour(img, label, color=(0,255,0)):
    _, contours, _ = cv2.findContours(label, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(img, contours, -1, color, 2, cv2.LINE_AA)
    return img



def compose_img(initial_img, img, α=0.8, β=1., λ=0.):
    '''
    img should be a blank image (all black) with region drawn on it.
    The result image is computed as follows:
        compose_img = initial_img * α + img * β + λ

    '''
    compose_img = cv2.addWeighted(initial_img, α, img, β, λ)
    return compose_img



def draw_segment_results1(img, label, prob):

    H,W,_ = img.shape
    coloring = np.zeros(shape=(H,W,3),dtype=np.uint8)
    coloring[:,:,0]= label
    coloring[:,:,2]= prob
    img = compose_img(img,coloring)

    return img



def draw_segment_results(img, label, prob):

    H, W, _ = img.shape

    coloring = np.zeros(shape=(H,W,3),dtype=np.uint8)
    coloring[:,:,1]= prob
    coloring[:,:,2]= prob
    img = compose_img(img,coloring,α=1, β=0.6)
    img = draw_contour(img, label, color=(0,0,255))

    return img





# train data functions ----------

def force_4d(x):
    if x.ndim == 3:
        x = x[:, :, :, np.newaxis]
    return x

def generate_train_batch_next(datas, labels, index, n, batch_size):

    i = n*batch_size
    batch_datas  = datas [i:i+batch_size]
    batch_labels = labels[i:i+batch_size]
    batch_index  = index [i:i+batch_size]
    return batch_datas, batch_labels, batch_index


def shuffle_data(datas, labels, index=None):

    num =len(datas)
    if index is None:
        index = list(range(num))

    random.shuffle(index)
    shuffle_datas  = datas[index]
    shuffle_labels = labels[index]

    return shuffle_datas, shuffle_labels, index



#-------------------------------------------------------------------


