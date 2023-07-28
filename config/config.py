import json
import os


class Config(object):
    db_config = {}

    _config = {}

    def __init__(self):
        self._config = self._get_cache()
        if "db" in self._config.keys():
            self.db_config = self._config['db']
        else:
            raise Exception("'db' is not find in config.json!")

    def get_config(self, key: str):
        keys = key.split(".")
        config = self._config
        for kid in keys:
            config = config[kid]
        return config

    @staticmethod
    def _get_cache():
        try:
            with open(os.path.dirname(os.path.realpath(__file__)) + "/" + "config.json", "r", encoding="utf8") as fp:
                json_data = json.load(fp)
                fp.close()
                return json_data
        except Exception as e:
            print(e)
