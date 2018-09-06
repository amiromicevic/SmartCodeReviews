# SmartCodeReviews

DESIGN

Data pipeline

	-connect to Rally
	-extract data
	-transform and load data to data store (database or file)

Processing

	-extract the new unprocessed data from database
	-run semantic analysis and summary extraction
	-create the new word cloud and frequent word counts on all data
	-write the output to database

Insights

	-extract data
	-display metrics and charts on PowerBI/Excel
	-display metrics and charts on AgileDashboard web app



IMPLEMENTATION

Data pipeline 

	-custom C# executable runs daily powered by Windows PowerShell scheduler

Processing
	
	-runs after Data pipeline process successfully completes
	-it is the same custom C# executable that integrates via IronPython with Python code that loads semantic analysis model from file and runs it

Insights
	
	-PowerBI connects to datastore, extracts and filters data to create visual dashboards

Login
	-Simple text file login to begin with (Time stamp, error message trail)
