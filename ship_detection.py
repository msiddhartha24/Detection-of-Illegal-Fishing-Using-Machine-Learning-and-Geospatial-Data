def detect_ships_from_image(image_path):
    import json, os, sys, random
    import numpy as np
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.optimizers import SGD
    from PIL import Image
    from matplotlib import pyplot as plt

    # Load dataset
    with open(r'C:\Users\Siddu\OneDrive\Desktop\Mini Project\Data\shipsnet.json') as f:
        dataset = json.load(f)

    input_data = np.array(dataset['data']).astype('uint8')
    output_data = np.array(dataset['labels']).astype('uint8')
    n_spectrum = 3
    weight = height = 80
    X = input_data.reshape([-1, n_spectrum, weight, height])
    y = to_categorical(output_data, 2)
    indexes = np.arange(4000)
    np.random.shuffle(indexes)
    X_train = X[indexes].transpose([0, 2, 3, 1]) / 255.0
    y_train = y[indexes]

    # Load or train model
    if os.path.exists("ship_detection_model.h5"):
        model = load_model("ship_detection_model.h5")
    else:
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=(80, 80, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(32, (10, 10), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

        sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        model.fit(X_train, y_train, batch_size=32, epochs=16, validation_split=0.2, shuffle=True, verbose=0)
        model.save("ship_detection_model.h5")

    # Load uploaded image
    image = Image.open(image_path)
    pix = image.load()
    width, height = image.size

    picture_vector = []
    for chanel in range(3):
        for y in range(height):
            for x in range(width):
                picture_vector.append(pix[x, y][chanel])

    picture_tensor = np.array(picture_vector).astype('uint8').reshape([3, height, width]).transpose(1, 2, 0)
    picture_tensor = picture_tensor.transpose(2, 0, 1)

    def cutting(x, y):
        area_study = picture_tensor[:, y:y + 80, x:x + 80]
        area_study = area_study.transpose([1, 2, 0])
        area_study = np.expand_dims(area_study, axis=0)
        return area_study / 255.0

    def not_near(x, y, s, coordinates):
        return all(not (x + s > e[0][0] > x - s and y + s > e[0][1] > y - s) for e in coordinates)

    def show_ship(x, y, thickness=5):
        for i in range(80):
            for ch in range(3):
                for th in range(thickness):
                    x_index = min(picture_tensor.shape[2] - 1, max(0, x - th))
                    if y + i < picture_tensor.shape[1]:
                        picture_tensor[ch][y + i][x_index] = 0
                    x_index = min(picture_tensor.shape[2] - 1, max(0, x + th + 80))
                    if y + i < picture_tensor.shape[1]:
                        picture_tensor[ch][y + i][x_index] = 0

        for i in range(80):
            for ch in range(3):
                for th in range(thickness):
                    y_index = min(picture_tensor.shape[1] - 1, max(0, y - th))
                    if x + i < picture_tensor.shape[2]:
                        picture_tensor[ch][y_index][x + i] = 0
                    y_index = min(picture_tensor.shape[1] - 1, max(0, y + th + 80))
                    if x + i < picture_tensor.shape[2]:
                        picture_tensor[ch][y_index][x + i] = 0

    # Detect ships
    step = 10
    coordinates = []
    batch_areas = []
    batch_positions = []

    for y in range(0, height - 80, step):
        for x in range(0, width - 80, step):
            area = cutting(x, y)
            batch_areas.append(area)
            batch_positions.append((x, y))

            if len(batch_areas) >= 32:
                batch_np = np.vstack(batch_areas)
                results = model.predict(batch_np, batch_size=32)

                for i, result in enumerate(results):
                    if result[1] > 0.90 and not_near(batch_positions[i][0], batch_positions[i][1], 88, coordinates):
                        coordinates.append([[batch_positions[i][0], batch_positions[i][1]], result])

                batch_areas = []
                batch_positions = []

    if batch_areas:
        batch_np = np.vstack(batch_areas)
        results = model.predict(batch_np, batch_size=32)
        for i, result in enumerate(results):
            if result[1] > 0.90 and not_near(batch_positions[i][0], batch_positions[i][1], 88, coordinates):
                coordinates.append([[batch_positions[i][0], batch_positions[i][1]], result])

    for e in coordinates:
        show_ship(e[0][0], e[0][1])

    picture_tensor_final = picture_tensor.transpose(1, 2, 0)
    plt.figure(figsize=(15, 30))
    plt.imshow(picture_tensor_final)
    plt.axis('off')
    plt.title("Detected Ships from Satellite Image")

    output_path = "detected_ships.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return output_path
