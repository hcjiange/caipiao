from com.base_model import BaseModel


class SingleCountModel(BaseModel):

    table = "single_count"
    database = "caipiao"
    
    def __init__(self):
        super().__init__(table=self.table, database=self.database)
