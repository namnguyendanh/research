from pairs import verify
import time


face_pairs = [
    ["faces/ldp_1.jpg", "faces/ldp_2.jpeg"],
    ["faces/ldp_2.jpg", "faces/ldp_3.jpg"]
]


start_time = time.time()
for face_pair in face_pairs:
    image_1 = face_pair[0]
    image_2 = face_pair[1]
    verify(image_1, image_2)
end_time = time.time()

print(f"{end_time - start_time}")