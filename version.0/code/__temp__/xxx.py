import glob
list =  glob.glob('/root/share/data/LUNA2016/dummy/npy/image/*.npy')

for i, name in enumerate(list):

    name = name.replace('/root/share/data/LUNA2016/dummy/npy/image/', '')
    name = name.replace('.npy', '')
    print (i, name)

    images = np.load('/root/share/data/LUNA2016/dummy/npy/image/' + name + '.npy')
    img = data_to_gray_img(np.squeeze(images[1]), is_auto=True)
    cv2.imwrite('/root/share/data/LUNA2016/dummy1/images/%04d.png' % i, img)

    images = np.load('/root/share/data/LUNA2016/dummy/npy/nodule_mask/' + name + '.npy')
    label = data_to_gray_img(np.squeeze(images[1]), is_auto=True)
    cv2.imwrite('/root/share/data/LUNA2016/dummy1/masks/%04d.png' % i, label)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = draw_contour(img, label)
    cv2.imwrite('/root/share/data/LUNA2016/dummy1/marked/%04d.png' % i, img)

    imshow('img', img)
    cv2.waitKey(1)

