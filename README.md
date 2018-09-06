# SmartCodeReviews

This project downloads team code reviews notes and applies NLP processing to the text entries, including semantic analysis and summary extraction. It also performs some more basic NLP opearations, such as frequent word counts and the creation of a word cloud.

The point is in the fact that a lot of "information" may be available to mine from the code review notes that can provide new, creative and yet undiscovered insights into individual and team's thinking, state, culture and areas for improvement.

Depending on the results of this project, particualrly of the Python componenet carrying out the NLP tasks, similar approach may be deployed to other areas that could benefit from natural text processing.

The project utilisies C# as the main service provider and Python as the data scientfic and machine learning component. The following section depicts the basic, overall design and implementation.

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


