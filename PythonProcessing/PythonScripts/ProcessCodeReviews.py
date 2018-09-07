# https://stackoverflow.com/questions/33725862/connecting-to-microsoft-sql-server-using-python
# this is work in progress

import clr
clr.AddReference('System.Data')
from System.Data.SqlClient import SqlConnection, SqlParameter

def getData():  
    newCodeReviews = list()
    
    connString = "N/A"
    connection = SqlConnection(connString)
    connection.Open()

    command = connection.CreateCommand()
    command.CommandText = 'SELECT * FROM CodeReview'

    reader = command.ExecuteReader()
    while reader.Read():
        newCodeReviews.append(reader['CodeReview'])

    connection.Close()

    return newCodeReviews

def processCodeReviews(newCodeReviews):
    # awaiting final code 

def run():
    newCodeReviews = getData()
    processCodeReviews(newCodeReviews)
