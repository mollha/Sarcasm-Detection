import os
import itertools
import xml.etree.ElementTree as ET

file_type = "Regular"
# file_type = "Ironic"

inputdir = "./" + file_type
csv = open(file_type + ".csv", "w+")


def parse_file(file_path: str):
    assert type(file_path) is str
    # get title, first index of _
    file_key = '"' + file_path[0:file_path.rfind('_')] + '"'
    file_path = os.path.join(inputdir, file_path)

    # --------- CONVERT AMPERSANDS -------------
    with open(file_path, 'r') as file:
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace(' & ', ' ampersand! ')
    filedata = filedata.replace('&', ' ampersand! ')
    filedata = filedata.replace('\n', ' ')


    # Write the file out again
    with open(file_path, 'w') as file:
        file.write(filedata)
    # ------------------------------------------

    csv_points = [file_key]
    with open(file_path, "r") as file:
        it = itertools.chain('<root>', file, '</root>')
        root = ET.fromstringlist(it)

        for item in [(root.find('.//STARS')), (root.find('.//TITLE')), (root.find('.//DATE')), (root.find('.//AUTHOR')),
                     (root.find('.//PRODUCT')), (root.find('.//REVIEW'))]:
            text = item.text
            text = text.replace('"', "'").strip('\n')
            text = text.replace('ampersand!', "&")
            text = text.replace('quot;', "'")
            csv_points.append('"' + text + '"')
    return ','.join(csv_points)


if __name__ == "__main__":
    count = 0
    csv.write("key,stars,title,date,author,product,review\n")
    for file_name in os.listdir(inputdir):
        if ".html" in file_name:
            continue
        print('FILE NAME ' + file_name +' -----------------------------')
        csv.write(parse_file(file_name) + '\n')