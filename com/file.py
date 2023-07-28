import json
import hashlib
import os


class File(object):

    # 读取json文件
    def read_json(self, symbol: str, type_dir: str):

        data = {}
        sub_dir = hashlib.md5(symbol[2:].encode('utf8')).hexdigest()[-2:]
        file_path = os.path.dirname(os.path.realpath(__file__)) + "/../../data/" + type_dir + "/" + sub_dir
        try:
            with open(file_path + "/" + symbol + ".json", "r", encoding="utf8") as fp:
                data = json.load(fp)
                fp.close()
        except Exception as e:
            print(e)

        return data

    # 写文件
    def write_file(self, symbol: str, type_dir: str, data: str):

        sub_dir = hashlib.md5(symbol[2:].encode('utf8')).hexdigest()[-2:]
        file_path = os.path.dirname(os.path.realpath(__file__)) + "/../data/" + type_dir + "/" + sub_dir
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        try:
            with open(file_path + "/" + symbol + ".json", "w+", encoding="utf8") as fp:
                fp.write(data + '\n')
        except Exception as e:
            print(e)
