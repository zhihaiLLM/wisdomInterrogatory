import requests
from requests.exceptions import RequestException

# 使用 Imgur API 创建一个匿名上传图片的请求
def upload_image_to_imgur(image_path):
    client_id = "YOUR_CLIENT_ID"
    headers = {"Authorization": f"Client-ID {client_id}"}
    try:
        with open(image_path, "rb") as image_file:
            response = requests.post("https://api.imgur.com/3/upload.json", headers=headers, files={"image": image_file})
            if response.status_code == 200:
                data = response.json()
                return data["data"]["link"]
            else:
                print("Failed to upload image to Imgur.")
                return None
    except RequestException as e:
        print(f"An error occurred: {str(e)}")
        return None

# 替换 YOUR_CLIENT_ID 为你的 Imgur 客户端 ID
image_url = upload_image_to_imgur("/root/data1/luwen/pics/logo.png")
if image_url:
    print(f"Image URL: {image_url}")
