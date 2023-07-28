import json
import re
import os
import sys
from common import Config


class Cache(object):

    _cache_type = ""

    def __init__(self):
        self._cache_type = Config().get_config("cache")

    # 读取缓存
    def read(self, key: str):

        data = ""
        if self._cache_type == "file":
            key = key.replace(":", "/")
            file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/cache"
            if not os.path.isfile(file_path + "/" + key + ".txt"):
                return ""

            try:
                with open(file_path + "/" + key + ".txt", "r", encoding="utf8") as fp:
                    data = fp.read()
                    fp.close()
            except Exception as e:
                print(e)
        if self._cache_type == "redis":
            pass

        return data

    # 设置缓存
    def set(self, key: str, data: (str, int, float, dict, list)):

        if self._cache_type == "file":
            key = "/" + key.replace(":", "/")
            file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/cache"
            if not os.path.exists(file_path + re.sub(r'/[^/]*$', "", key)):
                os.makedirs(file_path + re.sub(r'/[^/]*$', "", key))
            try:
                if type(data) == dict or type(data) == list:
                    data = json.dumps(data)
                with open(file_path + key + ".txt", "w+", encoding="utf8") as fp:
                    fp.write(str(data))
            except Exception as e:
                print(e)
                return False
        if self._cache_type == "redis":
            pass
        return True

    # 删除缓存
    def delete(self, key: str):

        if self._cache_type == "file":
            key = key.replace(":", "/")
            file_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/cache/" + key + ".txt"
            if os.path.exists(file_name):
                try:
                    os.remove(file_name)
                except Exception as e:
                    print(e)
        if self._cache_type == "redis":
            pass
