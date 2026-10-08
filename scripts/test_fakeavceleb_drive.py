import requests
import os

doc_ids = {
    "A_real_real.mp4": "1j2J0RCSTEHkzR7k-i9UT03vc4Xy2yFDG",
    "B_real_fakeaudio.mp4": "18nA84Yb0ZU5E7e3Z-sqe_8ZDGSYdAAUe",
    "C_fakevideo_realaudio.mp4": "1CG7F1n_B_mEHaXL6-ojwpJFa26HL92dL",
    "D_fakevideo_fakeaudio.mp4": "1TCKZAKhdgNJdze3njJHuOmOTaMf_XBld"
}

out_dir = "/tmp/fakeavceleb_samples"
os.makedirs(out_dir, exist_ok=True)

session = requests.Session()

for name, doc_id in doc_ids.items():
    url = f"https://drive.google.com/uc?id={doc_id}&export=download"
    print(f"Testing download for {name} ({doc_id})...")
    res = session.get(url, stream=True)
    print(f"Status: {res.status_code}, Content-Type: {res.headers.get('Content-Type')}")
    if res.status_code == 200:
        path = os.path.join(out_dir, name)
        with open(path, "wb") as f:
            for chunk in res.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
        size = os.path.getsize(path)
        print(f"Saved {name}: {size} bytes")
        # Check if it is actually an MP4 or HTML error
        with open(path, "rb") as f:
            head = f.read(16)
        print(f"Header: {head}")
    else:
        print("Failed.")
