import pymysql


class Mysql(object):
    config = None
    conn = None
    cursor = None

    def __init__(self, config):
        config["cursorclass"] = pymysql.cursors.DictCursor
        self.config = config
        self.connect()

    def __del__(self):
        self.close()

    def connect(self):
        self.conn = pymysql.connect(**self.config)
        self.cursor = self.conn.cursor()
        print("database is linking...")

    def close(self):
        self.cursor.close()
        self.conn.close()
        self.conn = None
        self.cursor = None
        print("database is close!")

    # ִ�е���sql
    def exec(self, sql, params=()):
        result = 0
        try:
            # ���ӶϿ�ʱ����������
            if self.conn is None:
                self.connect()
            # ִ��sql���
            result = self.cursor.execute(sql, params)
            # �ύ�����ݿ�ִ��
            self.conn.commit()
        except:
            # �ر����ݿ�����
            self.close()
            # �������������ع�
            self.conn.rollback()
        return result

    # ִ�е���sql
    def exec_all(self, sql_list, params=()):
        result = 0
        try:
            # ���ӶϿ�ʱ����������
            if self.conn is None:
                self.connect()
            # ִ��sql���
            for sql in sql_list:
                res = self.cursor.execute(sql, params)
                if res is not None:
                    result += 1
            # �ύ�����ݿ�ִ��
            self.conn.commit()
        except:
            # �ر����ݿ�����
            self.close()
            # �������������ع�
            self.conn.rollback()
        return result

    # ��ȡһ����¼
    def get_one(self, sql, params=()):
        result = None
        try:
            # ���ӶϿ�ʱ����������
            if self.conn is None:
                self.connect()
            self.cursor.execute(sql, params)
            result = self.cursor.fetchone()
        except Exception as e:
            print(e)
        return result

    # ��ȡ��������¼
    def get_many(self, sql, params=()):
        list_data = ()
        try:
            # ���ӶϿ�ʱ����������
            if self.conn is None:
                self.connect()
            self.cursor.execute(sql, params)
            list_data = self.cursor.fetchall()
        except Exception as e:
            print(e)
        return list_data
