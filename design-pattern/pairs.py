from deepface import DeepFace


def verify(image_1, image_2):
    model = DeepFace.build_model("Facenet")

    image_1 = model.predict(image_1)
    image_2 = model.predict(image_2)