import quandl
import os
from dotenv import load_dotenv

load_dotenv()  # Expects .env file containing QUANDL_API_KEY

quandl.ApiConfig.api_key = os.getenv('QUANDL_API_KEY')

# get WTI Crude Oil Price from the US Dept of Energy:
data = quandl.get("EIA/PET_RWTC_D")

# return data as numpy array:
data = quandl.get("EIA/PET_RWTC_D", returns="numpy")

# filter dates:
data = quandl.get("FRED/GDP", start_date="2001-12-31", end_date="2005-12-31")

# request specific columns
data = quandl.get(["NSE/OIL.1", "WIKI/AAPL.4"])

# request the last 5 rows:
data = quandl.get("WIKI/AAPL", rows=5)

# change the sampling frequency:
data = quandl.get("EIA/PET_RWTC_D", collapse="monthly")

# perform elementary calculations on the data:
data = quandl.get("FRED/GDP", transformation="rdiff")

# TABLES
# https://docs.data.nasdaq.com/docs/python-tables

# The examples below all involve the Mergent Global Fundamentals dataset, specifically the MER/F1 table. 
# This particular table is filterable on multiple columns, including compnumber, mapcode and reportdate. 
# This means that users can narrow down their request to rows with specific values for these (and all available) filters.

# NOTE:

# The tables API is limited to 10,000 rows per call. 
# However, when using the Python library, appending the argument paginate=True will extend the limit to 1,000,000 rows. 
# As such, we recommend using paginate=True for all calls. 
# Please note that some datasets can return more data than Python allows. 
# If this occurs, you will need to further filter your call to download less data, as outlined in the examples below. 
# Or you may consider using data exporter.

# Filter rows
# It is possible to download only certain desired rows from a table by 
# specifying one or more columns to act as criteria to filter rows. 
# If the value in a given column matches the filter argument, then the row containing that value is returned.

# Only columns designated as filterable in the table's documentation page can be used as criteria to filter rows.

# Download data for Nokia (compnumber=39102)
data = quandl.get_table('MER/F1', compnumber="39102", paginate=True)

# Download data for Nokia (compnumber=39102) and Deutsche Bank AG (compnumber=2438)
data = quandl.get_table('MER/F1', compnumber=["39102" , "2438"], paginate=True)

# Filter columns
# It is possible to select specific table columns to download by identifying them with the qopts.columns parameter.

# Download the compnumber column
data = quandl.get_table('MER/F1',qopts={"columns":"compnumber"}, paginate=True)

# Download the compnumber and ticker columns
data = quandl.get_table('MER/F1',qopts={"columns":["compnumber", "ticker"]}, paginate=True)

# Filter rows and columns

# Download the reportdate column for Nokia (compnumber=39102)
data = quandl.get_table('MER/F1',compnumber="39102", qopts={"columns":"reportdate"}, paginate=True)

# Download the reportdate, indicator, and amount columns for Nokia (compnumber=39102)
data = quandl.get_table('MER/F1',compnumber="39102", qopts={"columns":["compnumber", "ticker"]}, paginate=True)

# Download the closing prices for Apple (AAPL) and Microsoft (MSFT) between 2016-01-01 and 2016-12-31.
data = quandl.get_table('WIKI/PRICES', 
                        qopts = { 'columns': ['ticker', 'date', 'close'] }, 
                        ticker = ['AAPL', 'MSFT'], 
                        date = { 'gte': '2016-01-01', 'lte': '2016-12-31' })

# Download an entire table
# To retrieve table data:

quandl.get_table('MER/F1', paginate=True)

# This is the syntax for calling an entire table. 
# While most tables can be downloaded with such a call, MER/F1's size requires that you narrow down your request with filters, as shown above.

#Please note that this call returns a maximum of 1,000,000 rows. 
# To get more rows, you need to use the export_table function:

quandl.export_table('MER/F1')

# This call will save the data in a zip file called MER_F1.zip in your working directory. 
# You can specify the location of the downloaded zip file using the filename parameter:

quandl.export_table('MER/F1', filename='/my/path/db.zip')

# You can also export a subset of the data using filterable columns to filter on rows and the qopts parameter to filter on columns:

quandl.export_table('ZACKS/FC',  ticker=['AAPL', 'MSFT'], per_end_date={'gte': '2015-01-01'}, qopts={'columns':['ticker', 'per_end_date']})

# Depending on the size of the table, it may take a while to generate the zip file. 
# A message will be printed while the file is being generated. 
# After the file is generated and the download is finished, the file path of the downloaded zip file will be printed.