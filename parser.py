import os
# https://pypi.org/project/pyshp/
import shapefile as sf
 
import matplotlib.pyplot as plt
import numpy as np

# change region to process a different data set
region = "Gemeinde Schwerte"

# expect folder `foldername` relative to CWD containing files like `dataname`.prj, `dataname`.shp, `dataname`.shx, etc
foldername = "Testdaten/trainingdata/WFS NW ALKIS Grundrissdaten vereinfachtes Schema Regionen/"
foldername += region
dataname = "GebaeudeBauwerk"
 
# create a shapefile reader. Query data from this object
assert os.path.exists(foldername), "Input file does not exist."
file = sf.Reader(f"{foldername}/{dataname}")
print(file)
 
print("======= METADATA FIELDS 10 Entries ======")
print(list(map(lambda x: x[0], file.fields)))
for i in range(10):
    print(file.record(i))
print("======= METADATA FIELDS 10 Entries ======")
 
# sort data into lists of buildings containing each a list of polygons as numpy array with shape (poly_len,2)
def sort_into_shape_list(file: sf.Reader):
    # we use a dictonary of lagebezeichnung -> list<list<np.array>>
    lagebez_shapes = dict()
 
    for shapeRecord in file.shapeRecords():
        s: sf.ShapeRecord = shapeRecord # this is for intellisense, type hints in for do not seem to work
        bez = s.record['lagebeztxt'] # string, something like 'Blablastraße 22'
        if bez != "": # drop everything without a lagebez
 
            # init lists if this is the first entry for lagebez
            if bez not in lagebez_shapes:
                lagebez_shapes[bez] = list()
 
            # shape.parts are indices into shape.points for each polygon. Since a single polygon has parts [0] the last polygon is a special case.
            for i in range(len(s.shape.parts) - 1):
                lagebez_shapes[bez].append(np.array(s.shape.points[s.shape.parts[i]:s.shape.parts[i+1]]))
            lagebez_shapes[bez].append(np.array(s.shape.points[s.shape.parts[-1]:]))
 
    # drop all keys, we are only interested in the lists for each lagebez
    return list(lagebez_shapes.values())
 
shapes_list = sort_into_shape_list(file)
np.save(f"Testdaten/trainingdata/extracteddata/{region}", shapes_list)
 
# ====== PRINT ALL DATA USING PYPLOT =======
for shapes in shapes_list:
    for s in shapes:
        plt.plot(s[:,0], s[:,1], '.-')
plt.show()