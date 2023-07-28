from com.base_model import BaseModel


class HistoryModel(BaseModel):

    table = "history"
    database = "caipiao"
    
    def __init__(self):
        super().__init__(table=self.table, database=self.database)
