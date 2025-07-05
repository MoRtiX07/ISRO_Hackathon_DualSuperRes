import urllib.request

url = "https://upload.wikimedia.org/wikipedia/commons/2/2c/Satellite_image_of_India.jpg"
save_path = "../data/raw/image1.jpg"

try:
    urllib.request.urlretrieve(url, save_path)
    print("✅ Image downloaded and saved as image1.jpg")
except Exception as e:
    print("❌ Failed to download image:", e)
