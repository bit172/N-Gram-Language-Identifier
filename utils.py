import io


def read(file):
    """
    Reads all the lines of a file
    :param file: file name
    :return: list of all lines
    """
    f = io.open(file, "r", encoding="utf8")
    contents = f.readlines()
    f.close()
    return contents

