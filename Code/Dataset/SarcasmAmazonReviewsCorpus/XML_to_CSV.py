import os
import itertools
import xml.etree.ElementTree as ET

file_type = "Ironic"

inputdir = "./" + file_type
csv = open(file_type + ".csv","w+")

def parse_file(file_path: str):
    assert type(file_path) is str
    file_path = os.path.join(inputdir, file_path)
    csv_points = []
    with open(file_path, "r") as file:
        it = itertools.chain('<root>', file, '</root>')
        root = ET.fromstringlist(it)
        print(root.text)

        strings = (root.find('.//STARS'))
        csv_points.append('"' + strings.text + '"')

        strings = (root.find('.//TITLE'))
        csv_points.append('"' + strings.text + '"')
        print(csv_points)

        strings = (root.find('.//DATE'))
        csv_points.append('"' + strings.text + '"')
        print(csv_points)

        strings = (root.find('.//AUTHOR'))
        csv_points.append('"' + strings.text + '"')
        strings = (root.find('.//PRODUCT'))
        csv_points.append('"' + strings.text + '"')
        strings = (root.find('.//REVIEW'))
        csv_points.append('"' + strings.text + '"')
    return ','.join(csv_points)

if __name__ == "__main__":
    count = 0
    csv.write("stars,title,date,author,product,review\n")
    for file_name in os.listdir(inputdir):
        print('FILE NAME ' + file_name +' -----------------------------')
        if ".html" in file_name:
            continue
        csv.write(parse_file(file_name) + '\n')