import numpy as np
from PIL import Image, ImageDraw

# === FUNCTION TO RASTERIZE FLOOR PLATE ===
def raster_images(DIM, SAMPLES, REGION):
    # shapes is of the shape: np.array(list(np.array(np.array)))
    # to get a tuple use shapes[i][0][j][k], i=shape_nr, j=tuple_nr, k=tuple_idx
    shapes = np.load(f"Testdaten/trainingdata/extracteddata/{REGION}.npy", allow_pickle=True)

    # choose SAMPLES many random samples
    buildings = np.random.choice(shapes, SAMPLES)
    shapes = None

    # todo: sind haeuser ueberhaupt richtig? sind ja keine richtigen gebaeude wie in den daten
    # todo: ich male immer nur alles in einem polygon an, k.p. was mit den innenhoefen passiert

    # initialize 3D block for layers of 2D images
    im_as_np_array = np.zeros((len(buildings),) + DIM)
    for i in range(len(buildings)):
        max_val = buildings[i][0].max(axis=0)
        min_val = buildings[i][0].min(axis=0)

        # scale images to size DIM
        buildings[i][0] = (buildings[i][0] - min_val) / (max_val - min_val) * 256

        im = Image.new("L", DIM, 0)
        draw = ImageDraw.Draw(im)
        # https://stackoverflow.com/questions/10016352/convert-numpy-array-to-tuple
        draw.polygon(tuple(map(tuple, buildings[i][0])), fill=(255))

        # save image in 3D block
        im_as_np_array[i] = np.asarray(im)
    return im_as_np_array
# ====================