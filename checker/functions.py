import urllib
from urlparse import urlparse

# Downloads image via url
def downloadImage(url):
    opener = urllib2.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.137 Safari/537.36')]
    try:
        handle = opener.open(url)
        if (handle.getcode() == 200):
            image = np.asarray(bytearray(handle.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
            return image
    except:
        pass

    return None