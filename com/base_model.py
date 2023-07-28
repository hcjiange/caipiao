from com.mysql import Mysql
from config.config import Config


class BaseModel(Mysql):
    table = ""
    database = "caipiao"

    _where = []
    _data = []
    _order = ""
    _fields = "*"
    _limit = ""
    _last_sql = ""

    def __init__(self, database: str = "caipiao", table: str = ""):
        self.table = table
        self.database = database
        self._free()
        if database in Config().db_config.keys():
            db_config = Config().db_config[database]
            super().__init__(db_config)
        else:
            raise Exception("\"" + database + "\" is not find in db config from config.json!")

    def _free(self):
        self._where = []
        self._data = []
        self._order = ""
        self._fields = "*"
        self._limit = ""

    def where(self, where: (str, dict)):
        # 字符串
        if type(where) == str and where != "":
            where = where.replace("\"", "\\\"")
            self._where.append(where)
        # 字典
        if type(where) == dict and where is not None and where != {} and where != ():
            for key in where.keys():
                if type(where[key]) == list:
                    value = ""
                    if type(where[key][1]) != list:
                        value = "\"" + str(where[key][1]).replace("\"", "\\\"") + "\""
                    else:
                        for value_item in where[key][1]:
                            if value_item == "":
                                value_item += "\"" + str(value_item).replace("\"", "\\\"") + "\""
                            else:
                                value_item += ",\"" + str(value_item).replace("\"", "\\\"") + "\""
                        value = "(" + value + ")"
                    self._where.append(key + " " + str(where[key][0]) + " " + value)
                else:
                    value = "\"" + str(where[key]).replace("\"", "\\\"") + "\""
                    self._where.append(key + " = " + value)
        return self

    def where_in(self, key: str, data: list):
        if len(data) != 0:
            value = ""
            for value_item in data:
                if value_item == "":
                    value_item += "\"" + str(value_item).replace("\"", "\\\"") + "\""
                else:
                    value_item += ",\"" + str(value_item).replace("\"", "\\\"") + "\""
            value = "(" + value + ")"
            self._where.append(key + " IN " + value)
        return self

    def _build_where(self):
        where_str = ""
        for item in self._where:
            if where_str == "":
                where_str = item
            else:
                where_str += (" AND (" + item + ") ")
        return where_str

    def data(self, data: (dict, list)):
        self._data = data
        return self

    def fields(self, fields: str):
        self._fields = fields
        return self

    def order(self, order: str, kind: str = ""):
        self._order = order
        if kind != "":
            self._order += (" " + kind)
        return self

    def order_desc(self, order: str):
        self._order = order + " DESC"
        return self

    def order_asc(self, order: str):
        self._order = order + " ASC"
        return self

    def limit(self, limit: (str, int)):
        self._limit = str(limit)
        return self

    def page(self, page: int, page_size: int):
        self._limit = str((page - 1) * page_size) + "," + str(page_size)
        return self

    def get_last_sql(self):
        return self._last_sql

    # 操作

    def insert(self, data=None):
        if data is None:
            data = self._data
        if data == {}:
            return False

        keys = []
        values = []
        for key in data.keys():
            keys.append(str(key))
        for value in data.values():
            value = str(value if value is not None else "").replace("\"", "\\\"")
            values.append(value)

        sql_key = "(\"" + "\",\"".join(keys) + "\")"
        sql_value = "(\"" + "\",\"".join(values) + "\")"

        sql = "INSERT INTO " + str(self.database) + "." + str(self.table) + " " + str(sql_key) + " VALUES " + str(
            sql_value)

        result = super().exec(sql)
        self._last_sql = sql
        self._free()
        return result

    def save_all(self, data=None):

        if data is None:
            data = self._data
        if not data:
            return False

        sql_list = []
        for item in data:
            keys = []
            values = []
            updates = []
            for key in item.keys():
                keys.append(str(key))
                value = str((item[key] if not (item[key] == "None" or item[key] is None) else "")).replace("\"", "\\\"")
                updates.append(
                    key + "=\"" + value + "\"")
            for value in item.values():
                value = str((value if not (value == "None" or value is None) else "")).replace("\"", "\\\"")
                values.append(value)

            sql_key = "(" + ",".join(keys) + ")"
            sql_value = "(\"" + "\",\"".join(values) + "\")"
            sql_item = "INSERT INTO " + str(self.database) + "." + str(self.table) + " " + str(sql_key) + " VALUES " \
                       + str(sql_value) + "ON DUPLICATE KEY UPDATE " + ", ".join(updates) + ";"
            sql_list.append(sql_item)
            # print(sql_item)

        result = super().exec_all(sql_list)
        self._last_sql = sql_list[-1]
        self._free()
        return result

    def find(self, where="", fields: str = "*"):

        self.where(where)

        where = self._build_where()

        sql = "SELECT " + fields + " FROM " + self.database + "." + self.table + \
              ((" WHERE " + where) if where != "" else "") + " LIMIT 1"

        result = super().get_one(sql)
        self._last_sql = sql
        self._free()
        return result

    def max(self, field: str):

        where = self._build_where()

        sql = "SELECT MAX(" + field + ") AS " + field + " FROM " + self.database + "." + self.table + \
              ((" WHERE " + where) if where != "" else "") + " LIMIT 1"

        result = super().get_one(sql)
        self._last_sql = sql
        self._free()
        if field in result.keys():
            return result[field]
        return 0

    def all(self):

        where = self._build_where()

        sql = "SELECT " + self._fields + " FROM " + self.database + "." + self.table + \
              ((" WHERE " + where) if where != "" else "") + \
              ((" ORDER BY " + self._order) if self._order != "" else "") + \
              ((" LIMIT " + self._limit) if self._limit != "" else "")
        result = super().get_many(sql)
        self._last_sql = sql
        self._free()
        return result
